# 🚕 NCR Ride Booking Analytics

End-to-end data science project analysing **150,000 Delhi-NCR ride-booking records**
to understand cancellation patterns and build a predictive model.

## 🌐 Live Demo
👉 [View Interactive Dashboard] [vila-chung-ncr-analysis.streamlit.app](https://vila-chung-ncr-analysis.streamlit.app/)

## 📂 Project Structure
| Notebook | Description |
|---|---|
| 01_Data_Cleaning | Data wrangling, feature engineering, missing-value strategy |
| 02_EDA_Statistics | EDA, SQL analysis, Welch t-test, correlation heatmaps |
| 03_Data_Mining | Random Forest cancellation classifier (5-fold CV) |
| 04_Dashboard | Route network graph, interactive Plotly charts, insights |

## 🔍 Key Findings
- Overall booking completion rate: ~62%
- Peak-hour cancellation spikes identified via dual-axis time-series analysis
- Random Forest model trained with leakage-free preprocessing pipeline
- Underserved urban corridors identified with higher cancellation rates

## 🛠️ Tech Stack
Python · pandas · scikit-learn · Plotly · Seaborn · NetworkX · Streamlit · SQLite

## 📊 Data Source
[Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
· Kaggle · Author: Yash Devladdha · For educational purposes only.

## 👤 Author
Vila Chung · HKU BASc Social Data Science · 2025
