# KPI-Driven Clustering & Tier Optimization

This repository contains an end-to-end workflow for **customer segmentation and tier assignment**, combining classical clustering methods (Ward, KMeans) with **KPI-optimized policies** (SFI, PI, revenue). The project explores how unsupervised learning can be aligned with business objectives to create stable, meaningful tiers.

---

## 🔍 Project Overview

Traditional clustering methods (e.g., KMeans, Ward) optimize for geometric separation, **not business KPIs**.  
This repo investigates:

1. **Baseline clustering**
   - Ward hierarchical clustering
   - KMeans / KMeans++
   - Optional: HDBSCAN for density-based comparison

2. **Limitations of pure clustering**
   - Skewed tier distributions
   - High-revenue clusters do not always maximize PI
   - Silhouette vs KPI trade-offs
   - Instability when adjusting feature weights

3. **KPI-Driven Tier Assignment (Policy Layer)**
   Instead of forcing clustering to satisfy KPIs, we introduce a **policy layer** that reassigns tiers based on:
   - SFI (Strategic Focus Index)
   - PI (Performance Index)
   - Revenue contribution
   - Guardrails on TPA and tier size

---

## 🛠️ Features

- Ward K=3/4/5 clustering experiments  
- Silhouette analysis & stability diagnostics  
- KPI scoring model (Policy v2)  
- Automatic tier distribution with constraints  
- Feature engineering for PI, revenue, and cluster diagnostics  
- (Optional) HDBSCAN tests for irregular density  

---

## 📁 Repository Structure

