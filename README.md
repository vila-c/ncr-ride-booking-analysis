# 🚕 NCR Ride Booking Analytics

An end-to-end data science project analysing **150,000 Delhi-NCR ride-booking records** to understand cancellation patterns, build a predictive model, and surface mobility equity insights.

## 🌐 Live Demo
👉 [View Interactive Dashboard](https://vila-chung-ncr-analysis.streamlit.app/)

> Features an interactive **Cancellation Risk Predictor** — enter a booking's distance, fare, and timing to see the predicted cancellation risk and what's driving it.

---

## 🔄 CRISP-DM Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

| Phase | Notebook / File | Description |
|-------|----------------|-------------|
| Business Understanding | — | Identify NCR ride cancellation drivers and equity gaps |
| Data Understanding | `02_EDA` | EDA, SQL analysis, Welch t-test, correlation heatmaps |
| Data Preparation | `01_Cleaning` | Cleaning, feature engineering, missing value handling |
| Modelling | `03_Data_Mining` | Random Forest classifier, K-means risk clustering |
| Evaluation | `03_Data_Mining` | ROC-AUC, Precision-Recall Curve, F1, Confusion Matrix |
| Deployment | `app.py` | Streamlit interactive dashboard |

---

## 🤖 ML Pipeline

```
Raw Data (150,000 records)
        ↓
Data Cleaning & Feature Engineering  [01_Cleaning]
  · Missing value imputation
  · Datetime feature extraction (Hour, Weekday, Month)
  · Label encoding for categorical variables
        ↓
Exploratory Data Analysis            [02_EDA]
  · SQL-based analysis (SQLite)
  · Statistical testing (Welch t-test)
  · Correlation heatmaps
        ↓
Feature Selection (event-time only)  [03_Data_Mining]
  · Only features available at booking time
  · Removed post-hoc features (ratings, CTAT, VTAT)
        ↓
Train / Test Split
  · Stratified 75/25 split
  · Test-set imputation uses training medians only (leakage-free)
        ↓
Random Forest Classifier
  · class_weight = "balanced" (handles imbalanced classes)
  · 5-fold cross-validation
        ↓
Evaluation
  · ROC-AUC · Precision-Recall Curve · F1 · Confusion Matrix
        ↓
Streamlit Dashboard                  [app.py]
  · Interactive visualisations
  · Route network graph
  · Cancellation Risk Predictor
  · SQL Explorer
```

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `01_Data_Cleaning_and_Preparation.ipynb` | Data wrangling, feature engineering, missing-value strategy |
| `02_EDA_and_Statistics.ipynb` | EDA, SQL analysis, Welch t-test, correlation heatmaps |
| `03_Data_Mining_and_Patterns.ipynb` | Random Forest classifier, K-means clustering, full model evaluation |
| `04_Visualization_Dashboard_and_Insights.ipynb` | Route network graph, equity gap analysis, business insights |
| `app.py` | Streamlit interactive dashboard |

---

## 🔍 Key Findings

- **Overall completion rate: ~62%** — the remaining 38% of bookings fail, primarily due to no driver found or driver cancellation (supply-side problem)
- **Ride Distance is the strongest cancellation predictor (58% importance)** — longer rides carry significantly higher cancellation risk
- **Booking Value is the second strongest predictor (41% importance)** — higher-fare trips are more likely to be cancelled, possibly due to driver cherry-picking or passenger price sensitivity
- **Peak-hour cancellation rate is higher** — morning (7–9 AM) and evening (5–8 PM) rush hours show elevated cancellation rates due to supply-demand imbalance
- **Mobility equity gap identified** — peripheral pickup zones show cancellation rates up to ~1.5× the dataset average, raising fairness concerns for underserved communities

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| 5-fold CV ROC-AUC | 0.9725 ± 0.0007 |
| Test ROC-AUC | 0.97+ |
| Evaluation | Confusion Matrix · ROC Curve · Precision-Recall Curve |

> **Note on feature selection:** An earlier model using post-hoc features (driver ratings, customer ratings, CTAT) achieved AUC = 1.0 — this was identified as data leakage. The final model uses only features available at booking time, producing a realistic AUC of ~0.97.

---

## 🗺️ Mobility Equity Insight

A NetworkX route graph (Top 30 routes) visualises cancellation rates across corridors. A supplementary bar chart quantifies the **equity gap** — the difference in cancellation rates between the best and worst-served pickup zones — highlighting areas where platform optimisation for profitability may unintentionally disadvantage certain communities.

---

## ⚠️ Tool Suitability Notes

This project is built for **educational and portfolio purposes**. Some tools used here would require different choices in a production environment:

| Tool | Used For | Production Consideration |
|------|----------|--------------------------|
| Streamlit | Interactive dashboard | ✅ Rapid prototyping · ⚠️ Not designed for enterprise-scale traffic |
| SQLite (in-memory) | SQL workflow demonstration | ✅ Education · ⚠️ Use PostgreSQL / BigQuery in production |
| NetworkX | Route visualisation | ✅ Small graphs · ⚠️ Use a dedicated graph database (e.g. Neo4j) for large networks |
| K-means clustering | Risk segmentation concept | ✅ Concept demonstration · ⚠️ Consider DBSCAN or business-rule-based segmentation in production |
| scikit-learn Random Forest | Classification model | ✅ Education · ⚠️ Production pipelines typically use MLflow + model registry |
| pandas (in-memory) | Data processing | ✅ Up to ~1M rows · ⚠️ Use Spark / Dask for large-scale data |

---

## 🛠️ Tech Stack

Python · pandas · scikit-learn · Plotly · Seaborn · NetworkX · Streamlit · SQLite · scipy

---

## 📊 Data Source

[Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard) · Kaggle · Author: Yash Devladdha · For educational purposes only.

> This is a publicly shared dataset, not proprietary Uber data. It may contain simulated or anonymised records. Findings should not be generalised to real-world Uber operations.

---

## 👤 Author

Vila Chung · HKU BASc Social Data Science · 2025  
[GitHub](https://github.com/your-username/ncr-ride-booking-analysis)
