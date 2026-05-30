---
title: RetainX Customer Attrition Engine
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
pinned: false
license: mit
---

# RetainX: Real-Time Customer Attrition Engine

RetainX is an enterprise-grade predictive analytics dashboard and microservice designed to solve structural customer churn for global telecommunications operators. Powered by a dual-engine gradient boosting framework, the system flags high-risk subscribers in real time, enabling customer success teams to deploy targeted retention strategies before revenue leaks occur.

🌐 **Live Application UI**: [Hugging Face Space](https://huggingface.co/spaces/anuragn2107/Retainx-Churn-Prediction)  
🔌 **Programmatic API Documentation**: [FastAPI Docs Sandbox](https://anuragn2107-retainx-churn-prediction.hf.space/docs)

---

## 🏢 Business Case & Problem Statement

* **The Problem**: A telecommunications operator faces a structural churn rate of **~26.5%**. With Customer Acquisition Costs (CAC) scaling past \$150 per subscriber, keeping an existing user is 5x to 10x more cost-effective than acquiring a new one.
* **The Objective**: Maximize **Recall** (minimizing missed churners/False Negatives) and optimize **ROC-AUC** to isolate high-risk customer profiles without over-allocating financial loyalty incentives to stable accounts.
* **The Solution**: An interactive UI engine and REST API that ingests operational demographics, contract rules, and account financials to compute side-by-side risk matrices from production-tier Gradient Boosting frameworks.

---

## 🛠️ Technology Stack & Ecosystem Tools

The architecture is built entirely on a modern production-grade ML stack:

* **Programming Language**: `Python 3.13`
* **Data Manipulation & Processing**: `Pandas`, `NumPy`
* **Machine Learning Pipelines**: `Scikit-Learn` (Feature Scaling, Data Stratification, Evaluation Metrics)
* **Gradient Boosting Frameworks**: `XGBoost` (Level-wise Tree Expansion) & `LightGBM` (Leaf-wise Tree Expansion)
* **Model Serialization**: `Joblib`
* **Web UI Layer**: `Gradio 6.x`
* **Application API Gateway**: `FastAPI` (Natively exposed behind the Gradio container instance)
* **CI/CD & Cloud Infrastructure**: `GitHub Actions` (Automation workflow engine) & `Hugging Face Spaces` (Dockerized hosting platform)

---

## ⚔️ Algorithmic Showdown: XGBoost vs. LightGBM

The engine processes customer profiles through two distinct variations of Gradient Boosting:

1. **XGBoost (Level-wise Growth)**: Grows trees horizontally, balancing splits level by level. It provides excellent structural precision on medium-sized tabular frameworks and handles class imbalance natively via matrix weight configurations (`scale_pos_weight=3`).
2. **LightGBM (Leaf-wise Growth)**: Grows trees vertically by splitting nodes with maximum loss reduction. Utilizing *Gradient-based One-Side Sampling (GOSS)* and *Exclusive Feature Bundling (EFB)*, it delivers ultra-fast training velocities and minimal memory overhead.

---

## 📈 Methodology & Operational Pipeline

The project follows the standardized **CRISP-DM** framework:
1. **Data Ingestion & Cleaning**: Fixed string whitespace anomalies within continuous variables, cast features to proper types, and applied structural median imputations to handle missing indicators securely without data leakage.
2. **Feature Optimization**: Continuous metrics (`tenure`, `MonthlyCharges`, `TotalCharges`) are normalized via `StandardScaler`, and categorical fields undergo alignment transformations.
3. **Core Insights Matrix**: 
   * **High Risk Factors**: Month-to-month contracts, lack of online security/tech support add-ons, and high relative monthly expenditures.
   * **Retention Anchors**: Multi-year agreements, long customer tenure, and automatic credit card payment setups.

---

## 🚀 Future Roadmap

- [ ] **Explainable AI (XAI)**: Incorporate a `SHAP` (SHapley Additive exPlanations) dashboard block to visually illustrate exactly *which* feature pushed an account into a high-risk category.
- [ ] **Real-Time Streaming**: Connect the API backend to an active event streaming framework (e.g., Apache Kafka) to evaluate subscriber profiles automatically based on live app interactions.
- [ ] **Drift Monitoring**: Set up automated checks (using `Evidently AI`) to monitor if customer behavior changes over time, automatically triggering a model retrain if production performance degrades.

---

## 📄 License

This project is open-source and available under the **MIT License**.
