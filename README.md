# Architecting the Next-Generation Customer Tiering System  
### A KPI-Driven, Data-Integrated Architecture for Customer Tiering  
_Fusing Statistical Clustering, Semi-Supervision, and Policy-Aligned Optimization_

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

## 👥 Authors  

**UCLA Anderson MSBA — Class of 2025**

- Sailing Ni  
- Joy Yu  
- Peng Yang  
- Richard Sie  
- Yifei Wang

**Prepared for:**  
Microsoft MCAPS AI Transformation Group  

**Mentors:**  
Juhi Singh  
Bonnie Ao  

---



## ⚠️ Data Disclaimer  

The dataset used in this repository is a **hypothesized dataset created for academic experimentation and research purposes only**.  
It does **not** represent real Microsoft customer data and contains no actual customer, financial, or proprietary information.

---

## 📚 Future Publication  

A full written publication and technical report will be added here upon completion:  

**_→ [Publication Link — Coming Soon]_**

---

## 📂 Repository Structure  

root/
│
├── act1_natural_segmentation.ipynb
│ └─ Unsupervised clustering (Ward, K-Medoids, K-Means)
│
├── act2_semi_supervised_segmentation.ipynb
│ └─ Semi-supervised signals & cluster stabilization
│
├── act3_dynamic_tiering.ipynb
│ └─ KPI-driven ranking (Policy v2) & optional optimization
│
├── UCLA_Microsoft_Data.xlsx
│ └─ Hypothesized dataset for academic research
│
└── docs/
├── publication_draft/ (placeholder)
├── figures/
└── slides/



Each notebook maps to one layer of the system architecture:  
**Static Segmentation → Semi-Supervision → KPI/Optimization.**

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

---

## 📊 Key KPIs  

Our KPI suite ensures that segmentation is both data-valid and business-actionable:

- **TPA — Tier Potential Alignment**  
- **TCI_PI / TCI_REV — Tier Compactness Index**  
- **SFI — Strategic Focus Index**  



---

## 📄 License  

This repository is intended for academic, educational, and research use only.  
No real customer or proprietary Microsoft data is included.

---

## 🏢 Contact  

For questions or collaboration:  
**sailingni@ucla.edu**
