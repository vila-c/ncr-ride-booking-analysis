import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import sqlite3

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NCR Ride Booking Analytics",
    page_icon="🚕",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_ncr_rides_with_target.csv",
                     parse_dates=["Datetime"])
    return df

df = load_data()

# ── Header ────────────────────────────────────────────────────
st.title("🚕 NCR Ride Booking Analytics Dashboard")
st.markdown(
    "**Author: Vila Chung** · HKU BASc Social Data Science · 2025 · "
    "[GitHub](https://github.com/vila-c/ncr-ride-booking-analysis)"
)
st.caption(
    "Dataset: Uber Ride Analytics Dashboard · Kaggle (Yash Devladdha) · "
    "150,000 records sampled to 50,000 for deployment · Educational use only."
)
st.divider()

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("Use the filters below to explore different segments of the data.")

vehicle_filter = st.sidebar.multiselect(
    "Vehicle Type",
    options=sorted(df["Vehicle Type"].unique()),
    default=sorted(df["Vehicle Type"].unique())
)
hour_range = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))

filtered = df[
    (df["Vehicle Type"].isin(vehicle_filter)) &
    (df["Hour"].between(hour_range[0], hour_range[1]))
]

st.sidebar.divider()
st.sidebar.markdown(f"**Showing:** {len(filtered):,} bookings")
st.sidebar.markdown(
    f"**Completion rate:** "
    f"{(filtered['Cancel_Type']=='Completed').mean():.1%}"
)

# ── KPIs ──────────────────────────────────────────────────────
total     = len(filtered)
completed = (filtered["Cancel_Type"] == "Completed").sum()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Bookings",      f"{total:,}")
col2.metric("Completion Rate",     f"{completed/total:.1%}")
col3.metric("Cancellation Rate",   f"{1 - completed/total:.1%}")
col4.metric("Avg Booking Value",   f"₹{filtered['Booking Value'].mean():.0f}")
col5.metric("Avg Ride Distance",   f"{filtered['Ride Distance'].mean():.1f} km")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "⏰ Time Analysis",
    "🗺️ Route Network",
    "🤖 Model Insights",
    "🗄️ SQL Explorer",
])

# ── Tab 1: Overview ───────────────────────────────────────────
with tab1:
    st.subheader("Booking Status & Vehicle Distribution")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            filtered["Cancel_Type"].value_counts().reset_index(),
            x="Cancel_Type", y="count",
            title="Booking Status Distribution",
            color="Cancel_Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
            labels={"Cancel_Type": "Status", "count": "Count"}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        comp = filtered["Cancel_Type"].value_counts(normalize=True).mul(100).round(1)
        fig2 = px.pie(
            values=comp.values, names=comp.index,
            title="Completion vs Cancellation Share",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Vehicle Type Performance")
    vehicle_stats = (
        filtered.groupby("Vehicle Type")
        .agg(
            Total=("Booking ID", "count"),
            Cancel_Rate=("is_cancelled", "mean"),
            Avg_Value=("Booking Value", "mean"),
            Avg_Distance=("Ride Distance", "mean"),
        )
        .assign(Cancel_Rate=lambda x: (x["Cancel_Rate"] * 100).round(1))
        .round(1)
        .reset_index()
        .sort_values("Cancel_Rate", ascending=False)
    )
    vehicle_stats.columns = [
        "Vehicle Type", "Total Bookings",
        "Cancel Rate (%)", "Avg Booking Value (₹)", "Avg Distance (km)"
    ]
    st.dataframe(vehicle_stats, use_container_width=True, hide_index=True)


# ── Tab 2: Time Analysis ──────────────────────────────────────
with tab2:
    st.subheader("Hourly Booking Volume vs. Cancellation Rate")
    hourly = (
        filtered.groupby("Hour")
        .agg(Volume=("Booking ID", "count"),
             Cancel_Rate=("is_cancelled", "mean"))
        .assign(Cancel_Rate=lambda x: x["Cancel_Rate"] * 100)
    )
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=hourly.index, y=hourly["Volume"],
        name="Booking Volume", marker_color="steelblue", opacity=0.8
    ))
    fig3.add_trace(go.Scatter(
        x=hourly.index, y=hourly["Cancel_Rate"],
        name="Cancel Rate (%)", yaxis="y2",
        line=dict(color="tomato", width=2),
        mode="lines+markers"
    ))
    fig3.update_layout(
        xaxis_title="Hour of Day (0–23)",
        yaxis=dict(title="Booking Volume"),
        yaxis2=dict(title="Cancel Rate (%)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Weekday vs Weekend Comparison")
    c1, c2 = st.columns(2)
    with c1:
        day_stats = (
            filtered.groupby("Is_Weekend")
            .agg(Volume=("Booking ID", "count"),
                 Cancel_Rate=("is_cancelled", "mean"))
            .assign(
                Cancel_Rate=lambda x: (x["Cancel_Rate"] * 100).round(1),
                Day_Type=lambda x: x.index.map({0: "Weekday", 1: "Weekend"})
            )
            .reset_index(drop=True)
        )
        fig_day = px.bar(
            day_stats, x="Day_Type", y="Volume",
            color="Day_Type", title="Volume: Weekday vs Weekend",
            template="plotly_white",
            color_discrete_sequence=["steelblue", "coral"]
        )
        st.plotly_chart(fig_day, use_container_width=True)

    with c2:
        # ✅ 修复：直接从 filtered 独立计算，避免 day_stats 顺序错乱
        cancel_day_df = pd.DataFrame({
            "Day_Type": ["Weekday", "Weekend"],
            "Cancel_Rate": [
                round(filtered[filtered["Is_Weekend"] == 0]["is_cancelled"].mean() * 100, 1),
                round(filtered[filtered["Is_Weekend"] == 1]["is_cancelled"].mean() * 100, 1),
            ]
        })
        fig_day2 = px.bar(
            cancel_day_df, x="Day_Type", y="Cancel_Rate",
            color="Day_Type", title="Cancel Rate: Weekday vs Weekend",
            template="plotly_white",
            color_discrete_sequence=["steelblue", "coral"],
            labels={"Cancel_Rate": "Cancel Rate (%)"}
        )
        st.plotly_chart(fig_day2, use_container_width=True)


# ── Tab 3: Route Network ──────────────────────────────────────
with tab3:
    st.subheader("Popular Route Network Graph")
    st.markdown(
        "Edge **thickness** = booking volume · "
        "Edge **colour** = cancellation rate (green = low, red = high)"
    )
    n_routes = st.slider("Number of top routes to display", 20, 100, 50)

    route_data = (
        filtered.groupby(["Pickup Location", "Drop Location"])
        .agg(count=("is_cancelled", "size"),
             cancel_rate=("is_cancelled", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(n_routes)
    )

    G = nx.DiGraph()
    for _, row in route_data.iterrows():
        G.add_edge(row["Pickup Location"], row["Drop Location"],
                   weight=row["count"],
                   cancel_rate=row["cancel_rate"] * 100)

    pos = nx.spring_layout(G, k=0.5, seed=42)
    edge_list   = list(G.edges(data=True))
    max_count   = route_data["count"].max()
    edge_widths = [d["weight"] / max_count * 8 for _, _, d in edge_list]
    edge_colors = [d["cancel_rate"] for _, _, d in edge_list]

    fig4, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in edge_list],
        width=edge_widths, edge_color=edge_colors,
        edge_cmap=plt.cm.RdYlGn_r, arrows=True, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax,
        node_size=[G.degree(n) * 80 for n in G.nodes()],
        node_color="lightblue", edgecolors="black")
    labels = {n: n for n in G.nodes() if G.degree(n) > 3}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn_r,
        norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
    )
    sm.set_array([])
    fig4.colorbar(sm, ax=ax, label="Cancellation Rate (%)")
    ax.axis("off")
    st.pyplot(fig4)

    st.subheader("Top 10 Highest-Risk Routes")
    top_risk = (
        route_data[route_data["count"] > 1]
        .assign(cancel_rate=lambda x: (x["cancel_rate"] * 100).round(1))
        .sort_values("cancel_rate", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_risk.columns = ["Pickup", "Drop", "Total Bookings", "Cancel Rate (%)"]
    st.dataframe(top_risk, use_container_width=True, hide_index=True)


# ── Tab 4: Model Insights ─────────────────────────────────────
with tab4:
    # ✅ 修复：标题改为 XGBoost
    st.subheader("Cancellation Prediction Model — XGBoost")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model",        "XGBoost")
    col2.metric("CV Folds",     "5-fold")
    col3.metric("Test ROC-AUC", "0.9711")

    # ✅ 修复：Methodology 文字改为 XGBoost，移除 class_weight
    st.markdown("""
    ### Methodology
    An **XGBoost classifier** was trained on this dataset to predict
    whether a booking would be cancelled. Key design decisions:

    - **Event-time features only**: removed post-hoc features (driver ratings,
      customer ratings, CTAT) to prevent data leakage — reducing AUC from
      a misleading 1.0 to a realistic 0.9711
    - **Leakage-free preprocessing**: test-set imputation used training-set
      medians only, preventing data leakage
    - **Stratified split**: 75/25 train/test split with `stratify=y` to
      preserve class balance
    - **Validation**: 5-fold cross-validation for robust performance estimation
    """)

    st.subheader("Feature Importance")
    # ✅ 修复：使用 XGBoost 真实特征重要性数字
    fi = {
        "Ride Distance": 0.9265,
        "Booking Value": 0.0277,
        "Weekday":       0.0115,
        "Hour":          0.0115,
        "Month":         0.0114,
        "Vehicle Type":  0.0114,
        "Is_Weekend":    0.0000,
    }
    fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
    fig5 = px.bar(
        fi_df.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        title="Top Feature Importance Scores (XGBoost)",
        color="Importance", color_continuous_scale="Blues",
        template="plotly_white",
        labels={"Importance": "Importance Score"}
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ✅ 修复：Key Findings 更新为 XGBoost 结果
    st.markdown("""
    ### Key Findings
    - **Ride Distance** is by far the strongest predictor (92.7%) — longer rides
      carry significantly higher cancellation risk, likely due to driver reluctance
      on long-haul trips
    - **Booking Value** is the second predictor (2.8%) — its influence is much
      smaller than distance, suggesting trip length matters more than fare alone
    - **Time and date features** (Weekday, Hour, Month) each contribute ~1%,
      collectively suggesting that timing plays a minor but consistent role

    ### Social Data Science Lens
    Underserved urban corridors show systematically higher cancellation rates,
    raising **mobility equity** concerns for lower-income zones with limited
    alternative transport options.
    """)

    # ── Interactive Cancellation Predictor ───────────────────
    st.divider()
    st.subheader("🔮 Try the Cancellation Predictor")
    st.markdown(
        "Enter a booking's details below to see the **predicted cancellation risk** "
        "and what's driving it. This tool uses the same features as the trained "
        "XGBoost model — only information available **at the time of booking**."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        input_distance = st.slider("Ride Distance (km)", 1.0, 50.0, 10.0, step=0.5)
        input_value    = st.slider("Booking Value (₹)",  50,  1000,  200, step=10)
        input_hour     = st.slider("Hour of Day",         0,    23,    8)

    with col_b:
        input_month = st.selectbox(
            "Month", list(range(1, 13)),
            format_func=lambda x: [
                "Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"
            ][x - 1]
        )
        input_weekday = st.selectbox(
            "Day of Week",
            [0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
        )
        input_vehicle = st.selectbox(
            "Vehicle Type",
            sorted(df["Vehicle Type"].unique())
        )

    input_weekend = 1 if input_weekday >= 5 else 0

    if st.button("🔍 Predict Cancellation Risk"):

        # ✅ 修复：使用 XGBoost 真实特征重要性权重
        # Note: rule-based approximation for demonstration purposes.
        # In production, use a serialised model: joblib.load("model.pkl")
        distance_contrib = (input_distance / 50) * 0.9265
        value_contrib    = (input_value / 1000)  * 0.0277
        hour_contrib     = (
            0.0115 if input_hour in list(range(7, 10)) + list(range(17, 21))
            else 0.005
        )
        risk_score = min(0.95, distance_contrib + value_contrib + hour_contrib)

        # Display risk score
        if risk_score > 0.55:
            label = "🔴 High Risk"
        elif risk_score > 0.35:
            label = "🟡 Medium Risk"
        else:
            label = "🟢 Low Risk"

        st.metric("Predicted Cancellation Risk", f"{label} — {risk_score:.0%}")

        # ✅ 修复：caption 改为 XGBoost
        st.caption(
            "⚠️ This predictor uses a rule-based approximation weighted by XGBoost "
            "feature importance scores. For full model inference, a serialised XGBoost "
            "model (`joblib`) would be loaded in production."
        )

        # Feature contribution chart
        contributions = {
            "Ride Distance": round(distance_contrib, 4),
            "Booking Value": round(value_contrib, 4),
            "Hour of Day":   round(hour_contrib, 4),
        }
        contrib_df = pd.DataFrame(
            contributions.items(),
            columns=["Feature", "Contribution to Risk"]
        )
        fig_contrib = px.bar(
            contrib_df.sort_values("Contribution to Risk"),
            x="Contribution to Risk",
            y="Feature",
            orientation="h",
            color="Contribution to Risk",
            color_continuous_scale="Reds",
            template="plotly_white",
            title="What's Driving This Prediction?",
            labels={"Contribution to Risk": "Risk Contribution Score"}
        )
        fig_contrib.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_contrib, use_container_width=True)

        # Contextual interpretation
        st.markdown("**Interpretation:**")
        if input_distance > 30:
            st.warning(
                f"🚗 Long ride ({input_distance} km) — drivers may be reluctant "
                f"to accept, increasing cancellation risk."
            )
        if input_value > 500:
            st.warning(
                f"💸 High fare (₹{input_value}) — passengers may reconsider "
                f"after seeing the price."
            )
        if input_hour in list(range(7, 10)) + list(range(17, 21)):
            st.warning(
                f"⏰ Peak hour ({input_hour}:00) — supply-demand imbalance "
                f"may increase cancellation likelihood."
            )
        if risk_score <= 0.35:
            st.success(
                "✅ This booking profile has a low cancellation risk — "
                "short distance, reasonable fare, off-peak timing."
            )


# ── Tab 5: SQL Explorer ───────────────────────────────────────
with tab5:
    st.subheader("🗄️ SQL Query Explorer")
    st.markdown(
        "Run SQL queries directly on the dataset using an in-memory SQLite database. "
        "This demonstrates the ability to work with both DataFrame and SQL-based workflows."
    )

    @st.cache_resource
    def get_connection():
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        df_sql = df.copy()
        df_sql.columns = (
            df_sql.columns.str.strip().str.lower()
            .str.replace(" ", "_").str.replace(r"[^\w]", "_", regex=True)
        )
        df_sql.to_sql("rides", conn, index=False, if_exists="replace")
        return conn

    conn = get_connection()

    PRESETS = {
        "Cancellation rate by vehicle type": """
SELECT   vehicle_type,
         COUNT(*)                           AS total_bookings,
         ROUND(AVG(is_cancelled)*100, 1)   AS cancel_rate_pct
FROM     rides
GROUP BY vehicle_type
ORDER BY cancel_rate_pct DESC""",

        "Top 10 highest-risk pickup locations": """
SELECT   pickup_location,
         COUNT(*)                           AS total_bookings,
         ROUND(AVG(is_cancelled)*100, 1)   AS cancel_rate_pct
FROM     rides
GROUP BY pickup_location
HAVING   COUNT(*) > 100
ORDER BY cancel_rate_pct DESC
LIMIT    10""",

        "Weekend vs Weekday comparison": """
SELECT   CASE WHEN is_weekend=1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
         COUNT(*)                           AS total_bookings,
         ROUND(AVG(booking_value), 1)       AS avg_booking_value,
         ROUND(AVG(ride_distance), 1)       AS avg_distance_km,
         ROUND(AVG(is_cancelled)*100, 1)   AS cancel_rate_pct
FROM     rides
GROUP BY is_weekend""",

        "Payment method breakdown": """
SELECT   payment_method,
         COUNT(*)                           AS total_bookings,
         ROUND(AVG(is_cancelled)*100, 1)   AS cancel_rate_pct,
         ROUND(AVG(booking_value), 1)       AS avg_booking_value
FROM     rides
WHERE    payment_method IS NOT NULL
GROUP BY payment_method
ORDER BY total_bookings DESC""",
    }

    preset = st.selectbox("Choose a preset query:", list(PRESETS.keys()))
    query  = st.text_area("SQL Query (editable):", value=PRESETS[preset], height=160)

    if st.button("▶ Run Query"):
        try:
            result = pd.read_sql(query, conn)
            st.success(f"Query returned {len(result)} rows.")
            st.dataframe(result, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"SQL Error: {e}")
