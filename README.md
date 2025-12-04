# Architecting the Next-Generation Customer Tiering System  
### A KPI-Driven, Data-Integrated Architecture  
_Fusing Statistical Clustering • Semi-Supervision • Policy-Aligned Optimization_

---

## 📘 Project Overview  

This repository contains the full codebase and assets for the  
**UCLA Anderson MSBA × Microsoft MCAPS AI Transformation Capstone Project**.

Our objective is to design a **next-generation, data-driven customer tiering architecture** that integrates:

- **Static Statistical Clustering** (Ward, K-Medoids, K-Means)  
- **Semi-Supervised Refinement**  
- **KPI-Driven Tier Ranking Policies** (Policy v2)  
- **A Lightweight Optimization Layer** for coverage strategy  

The resulting framework is designed to be **principled, business-aligned, and operationally deployable**, replacing heuristic segmentation with a reproducible and scalable architecture.

---

🔗 Relevant Links

👉 Interactive LLM Agent Demo:
https://contoso-tier-guide.lovable.app/

👉 Full Codebase, Notebooks, and Publication Draft:
https://github.com/sni13/kpi-driven-clustering/

👉 Microsoft Tech Community Publication:
https://techcommunity.microsoft.com/blog/analyticsonazure/architecting-the-next-generation-customer-tiering-system/4475326


---

## ⚠️ Data Disclaimer  

The dataset used in this repository is a **hypothesized dataset created for academic experimentation and research purposes only**.  
It does **not** represent real Microsoft customer data and contains no actual customer, financial, or proprietary information.

---

## 👥 Authors & Mentors  

**UCLA Anderson MSBA — Class of 2025**

- Sailing Ni (sailing.ni.2025@ucla.edu)  
- Joy Yu (joy.yu.2025@anderson.ucla.edu)  
- Peng Yang  (peng.yang.2025@anderson.ucla.edu)  
- Richard Sie (richard.sie.2025@anderson.ucla.edu)  
- Yifei Wang (yifei.wang.2025@anderson.ucla.edu)

**Prepared for:**  
**Microsoft MCAPS AI Transformation**

**Mentors:**  
Juhi Singh — juhisingh@microsoft.com  
Bonnie Ao — ziqiaoao@microsoft.com  

---

## 🧩 System Architecture  

<img src="docs/solution_architecture.png" width="850">

---

## 📂 Repository Structure  

```text
root/
├── act1_natural_segmentation.ipynb
│   └── Unsupervised clustering (Ward, K-Medoids, K-Means)
├── act2_semi_supervised_segmentation.ipynb
│   └── Semi-supervised signals & cluster stabilization
├── act3_dynamic_tiering.ipynb
│   └── KPI-driven ranking (Policy v2) & optional optimization
├── UCLA_Microsoft_Data.xlsx
│   └── Hypothesized dataset for academic research  
├── llm/
│   ├── llm.py                 # LLM orchestrator logic
│   ├── render.yaml            # Deployment config (Azure/Render)
│   ├── requirements.txt       # Python dependencies
│   ├── start.sh               # Entry point script
│   ├── README.md              # LLM usage & instructions
│   └── __pycache__/           # Auto-generated cache
└── docs/
    ├── publication_draft/     # Placeholder for final paper
    ├── figures/
    └── slides/
```
---


## 📊 Key KPIs  

Our KPI suite ensures that segmentation is both data-valid and business-actionable:

- **TPA — Tier Potential Alignment**  
- **TCI_PI / TCI_REV — Tier Compactness Index**  
- **SFI — Strategic Focus Index**  


---


## 🧠 Methodology Summary  

### **1. Static Segmentation (Act 1)**  
- Engineered features for customer scale and potential  
- Compared clustering methods using a unified KPI framework  
- Evaluated with TPA, TCI (PI & Revenue), SFI  

### **2. Semi-Supervised Refinement (Act 2)**  
- Integrated natural clusters with business-informed heuristics  
- Resolved boundary accounts  
- Improved stability and interpretability  

### **3. KPI-Driven Re-Ranking & Optimization (Act 3)**  
- Implemented Policy v2 (PI × Revenue weighted)  
- Balanced statistical purity with real-world business needs  
- Designed lightweight resource allocation logic
- 
---


## 📄 License  

This repository is intended for academic, educational, and research use only.  
No real customer or proprietary Microsoft data is included.


