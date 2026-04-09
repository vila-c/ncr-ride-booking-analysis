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
    "[GitHub](https://github.com/[your-username]/ncr-ride-booking-analysis)"
)
st.caption(
    "Dataset: Uber Ride Analytics Dashboard · Kaggle (Yash Devladdha) · "
    "150,000 records sampled to 50,000 for deployment · Educational use only."
)
st.divider()

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.header("🔍 Filters")
st.sidebar.markdown(
    "**How to use these filters:**\n\n"
    "**Step 1.** Select one or more **Vehicle Types** below to focus on "
    "specific ride categories (e.g. Auto, Go Sedan). Removing a type hides "
    "its bookings from all charts.\n\n"
    "**Step 2.** Drag the **Hour of Day** slider to narrow the time window "
    "(e.g. set 7–9 to see morning rush only).\n\n"
    "All five tabs update automatically as you change filters."
)
st.sidebar.markdown("---")

vehicle_filter = st.sidebar.multiselect(
    "① Vehicle Type",
    options=sorted(df["Vehicle Type"].unique()),
    default=sorted(df["Vehicle Type"].unique()),
    help="Select which vehicle types to include. All are selected by default."
)
hour_range = st.sidebar.slider(
    "② Hour of Day", 0, 23, (0, 23),
    help="Filter bookings by hour. Drag endpoints to narrow the range."
)

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

    st.info(
        "**Reading these charts:** The bar chart (left) shows the raw count of each "
        "booking outcome. The pie chart (right) shows the same data as percentages. "
        "The largest slice is **Completed** (~62%), meaning about 4 in 10 bookings "
        "fail — mostly because no driver was found or the driver cancelled."
    )

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

    st.caption(
        "**Takeaway:** Cancellation rates are nearly identical across all vehicle "
        "types (37%–39%), which means the type of car you book does not meaningfully "
        "affect whether the ride gets cancelled."
    )


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

    st.info(
        "**Reading this chart:** The blue bars show how many bookings occur at each "
        "hour (left axis). The red line shows the cancellation rate at each hour "
        "(right axis). Booking volume peaks around 9–11 AM, but the cancellation "
        "rate stays remarkably flat (~38%) throughout the day, meaning time of day "
        "alone has almost no effect on whether your ride gets cancelled."
    )

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
        fig_day2 = px.bar(
            day_stats, x="Day_Type", y="Cancel_Rate",
            color="Day_Type", title="Cancel Rate: Weekday vs Weekend",
            template="plotly_white",
            color_discrete_sequence=["steelblue", "coral"],
            labels={"Cancel_Rate": "Cancel Rate (%)"}
        )
        st.plotly_chart(fig_day2, use_container_width=True)

    st.caption(
        "**Takeaway:** Weekday and weekend booking volumes and cancellation rates "
        "are almost identical, confirming that the day of the week has no significant "
        "impact on cancellations."
    )


# ── Tab 3: Route Network ──────────────────────────────────────
with tab3:
    st.subheader("Popular Route Network Graph")
    st.markdown(
        "This graph shows the most popular routes in the NCR ride network. "
        "Use the slider below to control how many routes are shown."
    )
    st.markdown(
        "Edge **thickness** = booking volume · "
        "Edge **colour** = cancellation rate (green = low, red = high)"
    )
    n_routes = st.slider(
        "Number of top routes to display", 20, 100, 50,
        help="Drag to show more or fewer routes. Start with 30-50 for a clear picture."
    )

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

    st.info(
        "**Reading this graph:** Each circle is a pickup or drop location. Lines "
        "between them represent routes. Thicker lines = more bookings on that route. "
        "**Green lines** = low cancellation rates (good service). **Red lines** = "
        "high cancellation rates (underserved routes). Larger circles indicate "
        "locations that appear in many routes."
    )

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

    st.caption(
        "**Takeaway:** Routes connecting peripheral areas (Noida Sector, Gurgaon "
        "Sector) to the city centre tend to have the highest cancellation rates, "
        "suggesting drivers are less willing to serve these longer, less profitable corridors."
    )


# ── Tab 4: Model Insights ─────────────────────────────────────
with tab4:
    st.subheader("Cancellation Prediction Model — XGBoost")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model",         "XGBoost")
    col2.metric("CV Folds",      "5-fold")
    col3.metric("Evaluation",    "ROC-AUC")

    st.markdown("""
    ### Methodology
    An **XGBoost classifier** (n_estimators=200, max_depth=8, learning_rate=0.1)
    was trained on this dataset to predict whether a booking would be cancelled.
    Key design decisions:

    - **Leakage-free preprocessing**: test-set imputation used training-set
      medians only, preventing data leakage
    - **Stratified split**: 75/25 train/test split with `stratify=y` to
      preserve class balance
    - **Validation**: 5-fold cross-validation for robust performance estimation
    """)

    st.subheader("Feature Importance")
    fi = {
        "Ride Distance": 0.9265,
        "Booking Value": 0.0277,
        "Hour": 0.0115,
        "Month": 0.0114,
        "Weekday": 0.0115,
        "Vehicle Type": 0.0114,
        "Is_Weekend": 0.0000,
    }
    fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
    fig5 = px.bar(
        fi_df.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        title="Top Feature Importance Scores",
        color="Importance", color_continuous_scale="Blues",
        template="plotly_white",
        labels={"Importance": "Importance Score"}
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.info(
        "**Reading this chart:** Each bar shows how much a feature contributes to "
        "the model's prediction. **Ride Distance** dominates at 92.7%, meaning the "
        "model relies almost entirely on trip length to predict cancellations. "
        "Longer rides are far more likely to be cancelled. All other features "
        "(fare, time, vehicle type) contribute very little by comparison."
    )

    st.markdown("""
    ### Key Findings
    - **Ride Distance** is the dominant predictor (92.7%) — longer rides have
      significantly higher cancellation risk, possibly due to driver reluctance
    - **Booking Value** contributes only 2.8% — fare matters far less than
      distance alone
    - **Time and date features** (Hour, Month, Weekday) each contribute ~1%,
      suggesting cancellation is driven almost entirely by trip distance

    ### Social Data Science Lens
    Underserved urban corridors show systematically higher cancellation rates,
    raising **mobility equity** concerns for lower-income zones with limited
    alternative transport options.
    """)


# ── Tab 5: SQL Explorer ───────────────────────────────────────
with tab5:
    st.subheader("🗄️ SQL Query Explorer")
    st.markdown(
        "Run SQL queries directly on the dataset using an in-memory SQLite database. "
        "This demonstrates the ability to work with both DataFrame and SQL-based workflows."
    )

    st.markdown(
        "**How to use this tool:**\n\n"
        "**Step 1.** Pick a **preset query** from the dropdown below — each one "
        "answers a specific business question.\n\n"
        "**Step 2.** Review the SQL code in the text area. You can **edit it** to "
        "customise the analysis (e.g. change `LIMIT 10` to `LIMIT 20`).\n\n"
        "**Step 3.** Click the **Run Query** button to execute and see results.\n\n"
        "💡 *Tip: The table name is `rides`. Column names use snake_case "
        "(e.g. `vehicle_type`, `ride_distance`, `is_cancelled`).*"
    )

    # Load data into SQLite
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

    # Preset queries
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

    preset = st.selectbox(
        "① Choose a preset query:",
        list(PRESETS.keys()),
        help="Each preset answers a different business question about the dataset."
    )
    query  = st.text_area("② SQL Query (editable):", value=PRESETS[preset], height=160)

    if st.button("③ ▶ Run Query"):
        try:
            result = pd.read_sql(query, conn)
            st.success(f"Query returned {len(result)} rows.")
            st.dataframe(result, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"SQL Error: {e}")




