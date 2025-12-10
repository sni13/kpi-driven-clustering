from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import json
from openai import OpenAI
import numpy as np, random
import scipy.stats
import datetime as dt
import matplotlib.pyplot as plt
import re
from scipy.stats import spearmanr
import psutil, os
from dotenv import load_dotenv

np.random.seed(42)
random.seed(42)

# ---- helper: natural "Tier A" → "Tier Z" ordering ----
_TIER_RE = re.compile(r"^\s*Tier\s+([A-Z])\s*$")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # OR replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],           # MUST include OPTIONS
    allow_headers=["*"],
)

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# =========================================================
# LOAD DATA & RUN DEFAULT
# =========================================================

filename = "../UCLA_Microsoft_Data.xlsx"
ContosoRevData = pd.read_excel(filename, sheet_name=1)
FortuneGlobal2000 = pd.read_excel(filename, sheet_name=2)
TAM = pd.read_excel(filename, sheet_name=3)

process = psutil.Process(os.getpid())
print("Memory Usage (MB):", process.memory_info().rss / (1024 * 1024))

# =========================================================
# 1. FEATURE ENGINEERING (no reassignment)
# =========================================================

def feature_engineering(df, tam):
    df = df.copy()

    # Step 0
    product_tam = tam[tam['Metric_Name'].str.contains("Proj_TAM_Prod")].copy()

    # Step 1
    geo_tam = (
        product_tam[product_tam['Attr_Name'] == 'Geo_Entity']
        .groupby('Attr_Value')['Metric_Value']
        .sum()
        .reset_index()
        .rename(columns={'Attr_Value': 'Geo_Entity', 'Metric_Value': 'Geo_TAM_FY30'})
    )

    cat_tam = (
        product_tam[product_tam['Attr_Name'] == 'Commercial_Category']
        .groupby('Attr_Value')['Metric_Value']
        .sum()
        .reset_index()
        .rename(columns={'Attr_Value': 'Commercial_Category', 'Metric_Value': 'Category_TAM_FY30'})
    )

    # Step 2
    df = df.merge(geo_tam, on='Geo_Entity', how='left')
    df = df.merge(cat_tam, on='Commercial_Category', how='left')

    # Step 3
    df['MarketShare_FY30_Geo'] = (
        df['PotentialRevenue_FY30'] / df['Geo_TAM_FY30']
    )

    df['MarketShare_FY30_Category'] = (
        df['PotentialRevenue_FY30'] / df['Category_TAM_FY30']
    )

    # Step 4 — CAGR
    df['Revenue_CAGR'] = (
        (df['TotalRevenue_FY26'] / df['TotalRevenue_FY22']) ** 0.25 - 1
    ) * 100

    # Step 5 — Potential
    df['Revenue_potential'] = (
        df['PotentialRevenue_FY30'] - df['TotalRevenue_FY26']
    )

    return df



# =========================================================
# 2. PI SCORING WITH VARIABLE WEIGHTS
# =========================================================

def calculate_pi_acct(df, weights):
    df = df.copy()

    w_runway = weights["w_runway"]
    w_growth = weights["w_growth"]
    w_geo    = weights["w_geo"]
    w_cat    = weights["w_cat"]

    sig_cols = [
        "Revenue_potential",
        "Revenue_CAGR",
        "MarketShare_FY30_Geo",
        "MarketShare_FY30_Category",
    ]

    Z = {}
    for c in sig_cols:
        s = df[c].astype(float)
        mu, sd = s.mean(), s.std(ddof=0)
        Z[c] = (s - mu) / (sd + 1e-12)

    df["PI_acct"] = (
        w_runway * Z["Revenue_potential"] +
        w_growth * Z["Revenue_CAGR"] +
        w_geo    * Z["MarketShare_FY30_Geo"] +
        w_cat    * Z["MarketShare_FY30_Category"]
    ).fillna(0.0)

    return df



# =========================================================
# 3. TIER DISTRIBUTION
# =========================================================

def calculate_tier_distribution(df):
    df = df.copy()

    tier_counts = df["MarketTier_FY26"].value_counts().sort_index()

    nA = tier_counts.get("Tier A", 0)
    nB = tier_counts.get("Tier B", 0)
    nC = tier_counts.get("Tier C", 0)
    nD = tier_counts.get("Tier D", 0)

    df_sorted = df.sort_values("PI_acct", ascending=False).reset_index(drop=True)

    raw = (
        ["Tier A"] * nA +
        ["Tier B"] * nB +
        ["Tier C"] * nC +
        ["Tier D"] * nD
    )

    df_sorted["PI_Tier_Raw"] = raw

    df = df.merge(df_sorted[["ID", "PI_Tier_Raw"]], on="ID", how="left")
    return df



# =========================================================
# 4. TIER CONSTRAINT
# =========================================================

def constrain_tier(orig, proposed):
    tier_to_int = {"Tier A":0, "Tier B":1, "Tier C":2, "Tier D":3}
    int_to_tier = {v:k for k,v in tier_to_int.items()}

    o = tier_to_int[orig]
    p = tier_to_int[proposed]

    allowed = [i for i in [o, o-1, o+1] if 0 <= i <= 3]

    if p in allowed:
        return int_to_tier[p]

    closest = min(allowed, key=lambda x: abs(x - p))
    return int_to_tier[closest]


def _natural_tier_order(labels):
    letters, others = [], []
    for lb in labels:
        m = _TIER_RE.match(str(lb))
        if m:
            letters.append((lb, m.group(1)))
        else:
            others.append(str(lb))

    if letters:
        letters_sorted = [lb for lb, _ in sorted(letters, key=lambda x: x[1])]
        return letters_sorted + sorted(others)

    return sorted(others)

# ---- helper: TCI calculation ----
def _tci_from_value(df: pd.DataFrame, tier_col: str, value_col: str) -> float:
    """TCI = 1 - WithinTierVar(X) / TotalVar(X)"""
    d = df[[tier_col, value_col]].dropna()
    if d.empty:
        return np.nan

    x = d[value_col].values
    tot_var = np.var(x, ddof=0)
    if tot_var <= 0:
        return 0.0

    N = len(d)
    wt_var = sum(
        (len(g) / N) * np.var(g[value_col].values, ddof=0)
        for _, g in d.groupby(tier_col)
    )

    tci = 1.0 - (wt_var / tot_var)
    return float(max(0.0, min(1.0, tci)))

# ---- main KPI computation ----
def compute_kpis(df, tier_col, pi_col, rev_col, sfi_param=2):
    """
    Compute Tier-Performance-Alignment (TPA), Tier-Compactness-Index (TCI),
    and Strategic-Focus-Index (SFI).
    """

    df = df.copy()

    # --- tier order ---
    unique_labels = list(df[tier_col].dropna().unique())
    ordered_labels = _natural_tier_order(unique_labels)

    # --- SFI ---
    tier_stats = (
        df.groupby(tier_col)
          .agg({pi_col: "mean", rev_col: "sum"})
          .rename(columns={pi_col: "pi_mean", rev_col: "rev_sum"})
    )

    total_rev = float(tier_stats["rev_sum"].sum() + 1e-12)
    L = max(0, int(sfi_param))

    strategic_tiers = ordered_labels[:L]
    sfi_rev = float(
        tier_stats.loc[
            tier_stats.index.isin(strategic_tiers), "rev_sum"
        ].sum()
    )
    SFI = sfi_rev / total_rev

    # --- TPA ---
    tier_rank_map = {
        label: rank for rank, label in enumerate(ordered_labels[::-1], start=1)
    }
    df["_tier_ordinal"] = df[tier_col].map(tier_rank_map).astype(float)

    valid = df[[pi_col, "_tier_ordinal"]].dropna()
    rho, _ = spearmanr(valid[pi_col], valid["_tier_ordinal"])
    TPA = float(rho)

    # --- TCI ---
    TCI_PI = _tci_from_value(df, tier_col=tier_col, value_col=pi_col)
    TCI_REV = _tci_from_value(df, tier_col=tier_col, value_col=rev_col)

    return {
        "TPA": round(TPA, 3),
        "TCI_PI": round(TCI_PI, 3),
        "TCI_REV": round(TCI_REV, 3),
        "SFI": round(SFI, 3),
        "Strategic_Tiers": strategic_tiers
    }

# ---- composite score for 4-phase optimizer ----
def composite_score(k):
    return (
        0.35 * k["TPA"] +
        0.35 * k["SFI"] +
        0.30 * (k["TCI_PI"] + k["TCI_REV"])
    )

import numpy as np
import pandas as pd


# ==========================================================
# PRETTY PRINT
# ==========================================================
def pretty_print(phase, it, score, k):
    print(
        f"[{phase}] Iter {it:<6d} | "
        f"Score={score:7.4f} | "
        f"TPA={k['TPA']:>5.3f} | "
        f"TCI_PI={k['TCI_PI']:>5.3f} | "
        f"TCI_REV={k['TCI_REV']:>5.3f} | "
        f"SFI={k['SFI']:>5.3f}"
    )


# ==========================================================
# Utility: Worst / Best per tier
# ==========================================================
def find_worst_best(df, tier_col, value_col):
    tiers = ["Tier A", "Tier B", "Tier C", "Tier D"]
    worst, best = {}, {}
    for t in tiers:
        sub = df[df[tier_col] == t]
        if len(sub) == 0:
            worst[t], best[t] = None, None
            continue
        worst[t] = sub[value_col].idxmin()
        best[t] = sub[value_col].idxmax()
    return worst, best


# ==========================================================
# Utility: Worst / Best POOL
# ==========================================================
def find_pool(df, tier_col, value_col, pool_k=10):
    tiers = ["Tier A", "Tier B", "Tier C", "Tier D"]
    worst_pool, best_pool = {}, {}

    for t in tiers:
        sub = df[df[tier_col] == t]
        if len(sub) == 0:
            worst_pool[t], best_pool[t] = [], []
            continue
        worst_pool[t] = list(sub.nsmallest(pool_k, value_col).index)
        best_pool[t] = list(sub.nlargest(pool_k, value_col).index)

    return worst_pool, best_pool



# ==========================================================
# ⭐️ FOUR-PHASE OPTIMIZER with PRETTY PRINT ⭐️
# ==========================================================
def optimize_tiers_four_phase(
        df,

        tier_col="MarketTier_FY26",
        pi_col="PI_acct",
        rev_col="TotalRevenue_FY26",
        new_col="ImprovedTier",

        max_iter_each_phase=2000,
        plateau_limit=500,

        pool_k=10,
        print_every=300
    ):

    df = df.copy()
    N = len(df)

    # Tier mappings
    valid_tiers = ["Tier A", "Tier B", "Tier C", "Tier D"]
    tier_to_level = {"Tier A":0, "Tier B":1, "Tier C":2, "Tier D":3}

    # Original tiers
    orig_arr = df[tier_col].astype(str).to_numpy()

    working_arr = np.array([
        t if t in valid_tiers else "Tier D"
        for t in orig_arr
    ])

    # Tier A/B counts fixed
    countA = np.sum(orig_arr == "Tier A")
    countB = np.sum(orig_arr == "Tier B")

    # ------------------------
    # KPI Evaluator
    # ------------------------
    def evaluate(a):
        df["__temp_tier"] = a
        k = compute_kpis(df, "__temp_tier", pi_col, rev_col)
        return composite_score(k), k


    # ------------------------
    # INITIAL
    # ------------------------
    best_score, best_k = evaluate(working_arr)
    best_arr = working_arr.copy()

    print("\n========== INITIAL STATE ==========")
    pretty_print("INIT", 0, best_score, best_k)
    print("===================================\n")


    # ==========================================================
    # helper: ±1 constraint
    # ==========================================================
    def violates(idx):
        return abs(tier_to_level[working_arr[idx]] - tier_to_level[orig_arr[idx]]) > 1


    # ==========================================================
    # helper: swap(i,j)
    # ==========================================================
    def try_swap(i, j):
        nonlocal best_score, best_k, best_arr, working_arr

        if i is None or j is None or i == j:
            return False

        old_i, old_j = working_arr[i], working_arr[j]
        working_arr[i], working_arr[j] = old_j, old_i

        # ±1 original tier constraint
        if violates(i) or violates(j):
            working_arr[i], working_arr[j] = old_i, old_j
            return False

        # Tier A/B count constraint
        if (np.sum(working_arr=="Tier A") != countA) or \
           (np.sum(working_arr=="Tier B") != countB):
            working_arr[i], working_arr[j] = old_i, old_j
            return False

        new_score, new_k = evaluate(working_arr)

        if new_score >= best_score:
            best_score, best_k = new_score, new_k
            best_arr = working_arr.copy()
            return True
        else:
            working_arr[i], working_arr[j] = old_i, old_j
            return False



    # ==========================================================
    # ⭐️ PHASE 1 — PI SINGLE SWAP
    # ==========================================================
    print("========== Phase 1: PI Single Swap ==========")
    plateau = 0

    for it in range(1, max_iter_each_phase+1):

        df["__curTier"] = working_arr
        worst, best = find_worst_best(df, "__curTier", pi_col)

        improved = False
        for upper, lower in [("Tier A","Tier B"), ("Tier B","Tier C"), ("Tier C","Tier D")]:
            if try_swap(worst[upper], best[lower]):
                improved = True
                break

        if improved:
            plateau = 0
        else:
            plateau += 1

        if it % print_every == 0:
            pretty_print("P1", it, best_score, best_k)

        if plateau >= plateau_limit:
            print(f"--- Plateau at iter {it}, moving to Phase 2 ---")
            break

    working_arr = best_arr.copy()



    # ==========================================================
    # ⭐️ PHASE 2 — PI POOL SWAP
    # ==========================================================
    print("\n========== Phase 2: PI Pool Swap ==========")
    plateau = 0

    for it in range(1, max_iter_each_phase+1):

        df["__curTier"] = working_arr
        worst_pool, best_pool = find_pool(df, "__curTier", pi_col, pool_k)

        improved = False

        for upper, lower in [("Tier A","Tier B"), ("Tier B","Tier C"), ("Tier C","Tier D")]:
            wp, bp = worst_pool[upper], best_pool[lower]
            if len(wp)==0 or len(bp)==0:
                continue

            i = np.random.choice(wp)
            j = np.random.choice(bp)

            if try_swap(i, j):
                improved = True
                break

        if improved:
            plateau = 0
        else:
            plateau += 1

        if it % print_every == 0:
            pretty_print("P2", it, best_score, best_k)

        if plateau >= plateau_limit:
            print(f"--- Plateau at iter {it}, moving to Phase 3 ---")
            break

    working_arr = best_arr.copy()



    # ==========================================================
    # ⭐️ PHASE 3 — Revenue SINGLE
    # ==========================================================
    print("\n========== Phase 3: Revenue Single Swap ==========")
    plateau = 0

    for it in range(1, max_iter_each_phase+1):

        df["__curTier"] = working_arr
        worst, best = find_worst_best(df, "__curTier", rev_col)

        improved = False
        for upper, lower in [("Tier A","Tier B"), ("Tier B","Tier C"), ("Tier C","Tier D")]:
            if try_swap(worst[upper], best[lower]):
                improved = True
                break

        if improved:
            plateau = 0
        else:
            plateau += 1

        if it % print_every == 0:
            pretty_print("P3", it, best_score, best_k)

        if plateau >= plateau_limit:
            print(f"--- Plateau at iter {it}, moving to Phase 4 ---")
            break

    working_arr = best_arr.copy()



    # ==========================================================
    # ⭐️ PHASE 4 — Revenue POOL
    # ==========================================================
    print("\n========== Phase 4: Revenue Pool Swap ==========")
    plateau = 0

    for it in range(1, max_iter_each_phase+1):

        df["__curTier"] = working_arr
        worst_pool, best_pool = find_pool(df, "__curTier", rev_col, pool_k)

        improved = False

        for upper, lower in [("Tier A","Tier B"), ("Tier B","Tier C"), ("Tier C","Tier D")]:
            wp, bp = worst_pool[upper], best_pool[lower]
            if len(wp)==0 or len(bp)==0:
                continue

            i = np.random.choice(wp)
            j = np.random.choice(bp)
            if try_swap(i, j):
                improved = True
                break

        if improved:
            plateau = 0
        else:
            plateau += 1

        if it % print_every == 0:
            pretty_print("P4", it, best_score, best_k)

        if plateau >= plateau_limit:
            print(f"--- Plateau at iter {it} ---")
            break



    # ==========================================================
    # FINAL
    # ==========================================================
    print("\n========== Optimization Completed ==========")
    pretty_print("FINAL", 0, best_score, best_k)

    df[new_col] = best_arr
    return df

# =========================================================
# 5. LLM WEIGHT REQUEST
# =========================================================

class UserPrompt(BaseModel):
    prompt: str

class ChatTurn(BaseModel):
    role: str
    content: str

class UserPromptWithHistory(BaseModel):
    prompt: str
    history: list[ChatTurn] | None = None


def get_llm_weights(user_prompt: str, history: list[dict] | None = None) -> dict:
    system_msg = """
    You are an expert in B2B segmentation and KPI modeling.
    
    Your role is to interpret the user's intent about segmentation, determine whether they
    want hybrid segmentation or KPI-driven segmentation, and assign appropriate KPI weights.
    
    You MUST return a JSON object with EXACTLY these fields:
    
    {
      "w_runway": <float>,
      "w_growth": <float>,
      "w_geo": <float>,
      "w_cat": <float>,
      "run_mode": "<none | hybrid | kpi>",
      "response_message": "<string>"
    }

    - You will receive the last few user/assistant messages as history.
    - You MUST treat that history as real memory.
    - If the user references something from earlier (e.g., “use the same weights as before”, 
      “increase geo from last time”, “run the same method again”), you MUST use the past information.
    - If the user's intent is unclear, you MUST ask a clarifying question inside 
      response_message while still outputting valid weights (weights must always sum to 1).
    - You may refer to context in response_message but NEVER include context in JSON.
    
    You MUST still obey all intent classification rules below.
    ------------------------------------------------------------
    INTENT CLASSIFICATION RULES
    ------------------------------------------------------------
    
    1. GENERAL CONVERSATION (No segmentation intent)
    - If the user is greeting, chatting, or asking non-segmentation questions:
        run_mode = "none"
        response_message = a normal helpful reply
        Weights must still be valid numbers that sum to 1.0.
    
    2. HYBRID SEGMENTATION (Default)
    - If the user expresses tiering or segmentation intent but does NOT mention KPI explicitly:
        run_mode = "hybrid"
    
    3. KPI-BASED SEGMENTATION (STRICT REQUIREMENT)
    - The user MUST explicitly and literally mention the letters “KPI”.
    - Valid triggers include:
          "optimize KPI"
          "maximize KPI"
          "use KPI"
          "prioritize KPI"
    - If “KPI” is NOT explicitly mentioned, do NOT use run_mode = "kpi".
    - Words like “prioritize”, “optimize”, “improve”, “weight”, or “focus on” by themselves 
      are NOT triggers.
      Examples of NON-triggers:
          "prioritize geolocation"
          "optimize growth"
          "focus on revenue potential"
    
    4. JSON STRICTNESS
    - The JSON MUST be strictly valid.
    - No additional commentary, no explanation outside the JSON object.
    - EVEN for general conversation, you MUST still return valid JSON using run_mode="none"
      and place the conversational reply inside response_message.
    """

    messages = [{"role": "system", "content": system_msg}]

    if history:
        for turn in history[-5:]:
            # ✅ Only include USER messages in history
            if turn["role"] == "user":
                messages.append({
                    "role": "user",
                    "content": turn["content"]
                })

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=messages
    )

    raw = response.choices[0].message.content.strip()

    # --- everything below stays exactly the same ---
    try:
        parsed = json.loads(raw)
    except:
        print("❌ JSON parsing failed! Raw LLM output:")
        print(raw)
        return {
            "w_runway": 0.40,
            "w_growth": 0.30,
            "w_geo":    0.15,
            "w_cat":    0.15,
            "run_mode": "none",
            "response_message": "Sorry, I didn’t understand that. Please rephrase."
        }

    required = ["w_runway", "w_growth", "w_geo", "w_cat",
                "run_mode", "response_message"]

    for k in required:
        if k not in parsed:
            if k.startswith("w_"):
                parsed[k] = {"w_runway": 0.40,
                             "w_growth": 0.30,
                             "w_geo": 0.15,
                             "w_cat": 0.15}[k]
            elif k == "run_mode":
                parsed[k] = "none"
            else:
                parsed[k] = ""

    w = [parsed["w_runway"], parsed["w_growth"], parsed["w_geo"], parsed["w_cat"]]
    total = sum(w)

    if total > 0:
        parsed["w_runway"] = w[0] / total
        parsed["w_growth"] = w[1] / total
        parsed["w_geo"]    = w[2] / total
        parsed["w_cat"]    = w[3] / total
    else:
        parsed["w_runway"] = 0.40
        parsed["w_growth"] = 0.30
        parsed["w_geo"]    = 0.15
        parsed["w_cat"]    = 0.15

    return parsed




# =========================================================
# 6. WEBHOOK — CALLS LLM + FEEDS INTO PI CALCULATION
# =========================================================

@app.post("/get_weights")
def webhook(req: UserPromptWithHistory):
    global ContosoRevData, TAM, FortuneGlobal2000

    process = psutil.Process(os.getpid())
    print("💾 Before endpoint work (MB):", process.memory_info().rss / (1024 * 1024))

    # ======================================================
    # 1. LLM → weights + segmentation choice
    # ======================================================
    weights_obj = get_llm_weights(
    user_prompt=req.prompt,
    history=[h.dict() for h in req.history] if req.history else None)
    print("\n⭐ RECEIVED FROM LLM ⭐")
    print(weights_obj)

    run_mode = weights_obj.get("run_mode", "none")
    user_reply = weights_obj.get("response_message", "")

    if run_mode == "none":
        print("💬 Conversation mode — no clustering executed.")
        return {"message": user_reply}
    
    # ======================================================
    # Create fresh working copies
    # ======================================================
    df = ContosoRevData.copy(deep=True)
    tam = TAM.copy(deep=True)

    # ======================================================
    # 2. Feature Engineering
    # ======================================================
    df = feature_engineering(df, tam)

    # ======================================================
    # 3. PI scoring (weights directly from LLM JSON)
    # ======================================================
    df = calculate_pi_acct(df, weights_obj)

    # ======================================================
    # 4. Segmentation Mode
    # ======================================================

    # ------------------------------------------------------
    # DEFAULT = HYBRID = 4-PHASE OPTIMIZER
    # ------------------------------------------------------
    if run_mode == "hybrid":
        print("🔵 Using HYBRID segmentation (DEFAULT: 4-phase optimizer).")

        df = optimize_tiers_four_phase(
            df,
            tier_col="MarketTier_FY26",
            pi_col="PI_acct",
            rev_col="TotalRevenue_FY26",
            new_col="ImprovedTier"
        )

        output_df = df[["ID", "ImprovedTier"]]

    # ------------------------------------------------------
    # KPI METHOD (OLD PI-BASED)
    # ------------------------------------------------------
    elif run_mode == "kpi":
        print("🟣 Using KPI-based segmentation.")

        # Rank by PI respecting original tier counts
        df = calculate_tier_distribution(df)

        # Apply ±1 constraint
        df["Tier_PI_Constrained"] = [
            constrain_tier(orig, raw)
            for orig, raw in zip(
                df["MarketTier_FY26"],
                df["PI_Tier_Raw"]
            )
        ]

        output_df = df[["ID", "Tier_PI_Constrained"]]

    # ======================================================
    # 5. Export output CSV
    # ======================================================
    process = psutil.Process(os.getpid())
    print("💾 After endpoint work (MB):", process.memory_info().rss / (1024 * 1024))

    # Convert dataframe to CSV
    stream = io.StringIO()
    output_df.to_csv(stream, index=False)
    stream.seek(0)

    filename = f"pi_output_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    weights_metadata = {
        "run_mode": run_mode,
        "w_runway": round(weights_obj["w_runway"], 4),
        "w_growth": round(weights_obj["w_growth"], 4),
        "w_geo": round(weights_obj["w_geo"], 4),
        "w_cat": round(weights_obj["w_cat"], 4)
    }
    
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Run-Mode": run_mode,
            "X-KPI-Weights": json.dumps(weights_metadata)
        }
    )

'''
uvicorn llm:app --reload --port 8000
cloudflared tunnel --url http://localhost:8000
'''
