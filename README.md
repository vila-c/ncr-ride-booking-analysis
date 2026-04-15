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
| Data Understanding | `01_Cleaning` · `02_EDA` | Initial exploration, SQL analysis, Welch t-test, correlation heatmaps |
| Data Preparation | `01_Cleaning` | Cleaning, feature engineering, missing value handling |
| Modelling | `03_Data_Mining` | XGBoost classifier |
| Evaluation | `03_Data_Mining` | ROC-AUC, Precision-Recall Curve, F1, Confusion Matrix |
| Deployment | `app.py` | Streamlit interactive dashboard |

> **Note on methodology ordering:** In this project, we intentionally run `01_Cleaning` before `02_EDA`. While CRISP-DM typically places Data Understanding before Data Preparation, our workflow reflects the iterative nature of the framework.
> - We first perform initial cleaning and basic checks in `01_Cleaning` to ensure data quality and consistency.
> - Then, we conduct extended exploratory analysis (EDA) in `02_EDA`, applying statistical tests and correlation analysis on the cleaned dataset.
>
> This ordering avoids misleading insights from raw, noisy data and highlights the iterative feedback loop between Data Understanding and Data Preparation.

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
  · Only features available at trip start
  · Removed post-hoc features (ratings used to define target, not as features)
  · Fixed cleaning-induced data leakage (AUC 0.97 → 0.56)
        ↓
Train / Test Split
  · Stratified 75/25 split
  · Test-set imputation uses training medians only (leakage-free)
        ↓
XGBoost Classifier
  · Target: needs_intervention (Incomplete + low-rated Completed)
  · 5-fold cross-validation
  · Evaluated on ROC-AUC, Precision-Recall, F1, Confusion Matrix
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
| `03_Data_Mining_and_Patterns.ipynb` | XGBoost classifier, full model evaluation |
| `04_Visualization_Dashboard_and_Insights.ipynb` | Route network graph, equity gap analysis, business insights |
| `app.py` | Streamlit interactive dashboard |

---

## 🔍 Key Findings

- **Overall completion rate: ~62%** — the remaining 38% of bookings are non-completed, primarily due to no driver found or driver cancellation (supply-side problem)
- **Data cleaning error discovered and fixed** — the original cleaning pipeline applied unconditional median imputation, filling 48,000 cancelled bookings with identical placeholder values (Distance = 23.72 km, Fare = ₹414). This has been corrected to preserve NaN for trips that never started
- **Original AUC 0.97 was a data artifact** — the model was detecting imputed values, not real cancellation patterns. After correction, AUC dropped to 0.56, confirming the original score was inflated
- **Corrected model predicts trip intervention needs** — target includes Incomplete trips (mid-journey failures) + Completed trips with low ratings (Driver or Customer Rating < 4.0)
- **Feature importance is evenly distributed (~14–25% each)** — no single feature dominates, meaning trip outcomes depend on factors not captured in this dataset (driver behaviour, vehicle condition, weather)
- **Time features (Hour, Weekday, Month) have negligible effect** — peak-hour hypothesis tested and not supported; timing has no meaningful impact on non-completion
- **Mobility equity gap identified** — peripheral pickup zones show non-completion rates up to ~1.2× the dataset average

---

## 📊 Model Performance

| Metric | Original (artifact) | Corrected |
|--------|--------------------:|----------:|
| Model | XGBoost | XGBoost |
| Target | `is_cancelled` | `needs_intervention` |
| 5-fold CV ROC-AUC | 0.9725 | 0.5593 |
| Test ROC-AUC | 0.9711 | 0.5600 |
| Avg Precision | 0.9646 | 0.4534 |

> **What happened:** The original cleaning pipeline applied unconditional median imputation to all rows, filling 48,000 cancelled bookings (trips that never started) with identical values (Distance = 23.72 km, Fare = ₹414). The original model achieved AUC 0.97 by detecting this imputation pattern — not real cancellation signals. After restricting imputation to only Completed + Incomplete orders (trips that actually started), we redefined the target as `needs_intervention` (Incomplete trips + Completed trips with either rating < 4.0). The corrected AUC of 0.56 reflects the honest predictive power of trip-start features. **An anomalously high AUC should always be investigated as a potential data leakage signal.**

---

## 🗺️ Mobility Equity Insight

A NetworkX route graph (Top 30 routes) visualises non-completion rates across corridors. A supplementary bar chart quantifies the **equity gap** — the difference in non-completion rates between the best and worst-served pickup zones — highlighting areas where platform optimisation for profitability may unintentionally disadvantage certain communities.

---

## ⚠️ Tool Suitability Notes

This project is built for **educational and portfolio purposes**. Some tools used here would require different choices in a production environment:

| Tool | Used For | Production Consideration |
|------|----------|--------------------------|
| Streamlit | Interactive dashboard | ✅ Rapid prototyping · ⚠️ Not designed for enterprise-scale traffic |
| SQLite (in-memory) | SQL workflow demonstration | ✅ Education · ⚠️ Use PostgreSQL / BigQuery in production |
| NetworkX | Route visualisation | ✅ Small graphs · ⚠️ Use a dedicated graph database (e.g. Neo4j) for large networks |
| scikit-learn / XGBoost | Classification model | ✅ Education · ⚠️ Production pipelines typically use MLflow + model registry |
| pandas (in-memory) | Data processing | ✅ Up to ~1M rows · ⚠️ Use Spark / Dask for large-scale data |

---

## 🛠️ Tech Stack

Python · pandas · scikit-learn · XGBoost · Plotly · Seaborn · NetworkX · Streamlit · SQLite · scipy

---

## 📊 Data Source

[Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard) · Kaggle · Author: Yash Devladdha · For educational purposes only.

> This is a publicly shared dataset, not proprietary Uber data. It may contain simulated or anonymised records. Findings should not be generalised to real-world Uber operations.

---

## 👤 Author

Vila Chung · HKU BASc Social Data Science · 2027
[GitHub](https://github.com/vila-c/ncr-ride-booking-analysis)
