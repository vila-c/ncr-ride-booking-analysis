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
col3.metric("Non-completion Rate",   f"{1 - completed/total:.1%}",
            help="Includes all non-completed bookings: Driver Cancelled, Customer Cancelled, No Driver Found, and Incomplete.")
col4.metric("Avg Passenger Fare",   f"₹{filtered['Booking Value'].mean():.0f}",
            help="Average fare for trips that actually started (Completed + Incomplete). "
                 "Cancelled/No Driver Found bookings have no fare data.")
col5.metric("Avg Ride Distance",   f"{filtered['Ride Distance'].mean():.1f} km",
            help="Average distance for trips that actually started. "
                 "NaN values (cancelled bookings) are excluded.")
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

    st.warning(
        "**Important note on data cleaning:** During our analysis, we discovered "
        "that the original cleaning pipeline (Notebook 01) applied **unconditional "
        "median imputation** to all rows — including 48,000 cancelled/no-driver "
        "bookings where Ride Distance, Fare, and Ratings were legitimately missing "
        "(the trip never started, so no real values exist). This produced identical "
        "placeholder values (Distance = 23.72 km, Fare = ₹414) for all non-started "
        "trips, which the original XGBoost model learned to detect (AUC 0.97). "
        "**We fixed this** by restricting imputation to only Completed and Incomplete "
        "orders (trips that actually started). The corrected model now predicts "
        "**trip intervention needs** (mid-trip failures + poor ratings) with an "
        "honest AUC of ~0.56, confirming that the available features have limited "
        "predictive power for trip outcomes. "
        "**Lesson learned: an anomalously high AUC (> 0.95) should always be "
        "investigated as a potential data leakage signal.**"
    )

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
        "**What these charts show:** The bar chart (left) counts how many bookings "
        "ended in each outcome. **Completed** is the tallest bar at ~93,000 bookings. "
        "The remaining ~57,000 bookings failed, split across four categories. "
        "The pie chart (right) shows the same data as percentages: only **62%** of "
        "rides actually completed. The biggest failure mode is **Driver Cancelled** "
        "(18%), followed by **No Driver Found** (7%) and **Customer Cancelled** (7%) "
        "— both supply-side problems where the platform could not match the passenger "
        "with a willing driver. **Incomplete** rides account for 6%."
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
        "Non-completion Rate (%)", "Avg Passenger Fare (₹)", "Avg Distance (km)"
    ]
    st.dataframe(vehicle_stats, use_container_width=True, hide_index=True)

    st.info(
        "**What this table shows:** All seven vehicle types have non-completion rates "
        "clustered tightly between 37% and 39%. This **near-identical spread is itself "
        "a key finding** — it means vehicle type has virtually no impact on whether a "
        "ride gets completed. Whether a passenger books an Auto (cheapest) or a Prime "
        "SUV (most expensive), the odds of non-completion are statistically the same. "
        "Note: Avg Passenger Fare and Avg Distance are calculated only from trips that "
        "actually started (Completed + Incomplete); cancelled bookings have no real "
        "distance or fare data."
    )


# ── Tab 2: Time Analysis ──────────────────────────────────────
with tab2:
    st.subheader("Hourly Booking Volume vs. Non-completion Rate")
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
        name="Non-completion Rate (%)", yaxis="y2",
        line=dict(color="tomato", width=2),
        mode="lines+markers"
    ))
    fig3.update_layout(
        xaxis_title="Hour of Day (0–23)",
        yaxis=dict(title="Booking Volume"),
        yaxis2=dict(title="Non-completion Rate (%)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "**What this chart shows:** This is a dual-axis chart. The **blue bars** "
        "(left axis) show how many bookings were placed at each hour. The **red "
        "line** (right axis) shows the non-completion rate at that hour.\n\n"
        "**Key finding:** Booking volume peaks during the evening rush (17–19 PM, "
        "with hour 18 the single highest at ~12,400 bookings), and a secondary "
        "peak around 10 AM, matching typical commuting patterns. However, the red "
        "non-completion line is almost perfectly flat at ~38% across all 24 hours. "
        "Whether someone books at 3 AM or 6 PM, the probability of non-completion "
        "is essentially the same. "
        "**Time of day alone does not drive non-completion in this dataset.**"
    )

    with st.expander("📋 Hypothesis validation: does peak hour have higher non-completion?"):
        st.markdown(
            "**Hypothesis:** Morning rush (7–9 AM) and evening rush (17–20 PM) "
            "should show higher non-completion rates because supply-demand "
            "imbalance is more severe during commuting hours."
        )
        def _time_bin(h):
            if h in (7, 8, 9): return "Morning Rush (7–9)"
            elif h in (10, 11, 12, 13): return "Midday (10–13)"
            elif h in (14, 15, 16): return "Afternoon (14–16)"
            elif h in (17, 18, 19, 20): return "Evening Rush (17–20)"
            elif h in (21, 22, 23): return "Night (21–23)"
            else: return "Late Night (0–6)"
        _tb = filtered.copy()
        _tb["Time Bin"] = _tb["Hour"].apply(_time_bin)
        _tb_stats = (
            _tb.groupby("Time Bin")
            .agg(Bookings=("Booking ID", "count"),
                 Rate=("is_cancelled", "mean"))
            .assign(Rate=lambda x: (x["Rate"] * 100).round(2))
            .sort_values("Rate", ascending=False)
            .reset_index()
        )
        _tb_stats.columns = ["Time Bin", "Bookings", "Non-completion Rate (%)"]
        st.dataframe(_tb_stats, use_container_width=True, hide_index=True)
        _range = _tb_stats["Non-completion Rate (%)"].max() - _tb_stats["Non-completion Rate (%)"].min()
        st.markdown(
            f"**Result:** The range across all time bins is just **{_range:.2f} "
            f"percentage points**. Morning Rush and Evening Rush do not show "
            f"meaningfully higher non-completion rates than off-peak hours. "
            f"**The hypothesis is not supported by this dataset** — time of day "
            f"has negligible impact on booking outcomes."
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
            color="Day_Type", title="Non-completion Rate: Weekday vs Weekend",
            template="plotly_white",
            color_discrete_sequence=["steelblue", "coral"],
            labels={"Cancel_Rate": "Non-completion Rate (%)"}
        )
        st.plotly_chart(fig_day2, use_container_width=True)

    st.info(
        "**What these charts show:** The left chart compares total booking volume "
        "between weekdays and weekends. The right chart compares their non-completion "
        "rates.\n\n"
        "**Key finding:** Both the volume split (~71% weekday / ~29% weekend) and the "
        "non-completion rates are nearly identical — weekday 38.1% vs weekend 37.7%, "
        "a difference of just 0.4 percentage points. **This is not a meaningful "
        "difference.** In real ride-hailing data, you would expect weekend patterns "
        "to differ noticeably (different trip types, different driver availability)."
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
        "Edge **colour** = non-completion rate (green = low, red = high)"
    )

    # ── Location classifier (used by expander and descriptions) ──
    _all_locs = sorted(set(
        filtered["Pickup Location"].unique().tolist()
        + filtered["Drop Location"].unique().tolist()
    ))

    def _classify_ncr(name):
        n = name.lower()
        if "noida" in n or n == "botanical garden" or n == "greater noida":
            return "Noida"
        if any(k in n for k in [
            "gurgaon", "dlf", "cyber hub", "golf course", "sushant lok",
            "udyog vihar", "iffco", "huda city", "sohna", "manesar",
            "palam vihar", "sikanderpur", "mg road", "ardee", "vatika",
            "badshahpur", "hero honda", "kherki daula", "basai", "khandsa",
            "pataudi", "gwal pahari", "subhash chowk", "ambience mall",
            "civil lines gurgaon", "old gurgaon", "sadar bazar gurgaon",
        ]):
            return "Gurgaon"
        if "faridabad" in n:
            return "Faridabad"
        if any(k in n for k in [
            "ghaziabad", "indirapuram", "vaishali", "kaushambi", "raj nagar",
        ]):
            return "Ghaziabad"
        if any(k in n for k in [
            "meerut", "panipat", "sonipat", "bhiwadi", "bahadurgarh",
        ]):
            return "Outer NCR"
        return "Delhi"

    _loc_map = {loc: _classify_ncr(loc) for loc in _all_locs}

    # ── Top 10 Popular Routes reference table ──
    _top_routes = (
        filtered.groupby(["Pickup Location", "Drop Location"])
        .agg(Bookings=("Booking ID", "count"),
             NonComp=("is_cancelled", "mean"))
        .assign(NonComp=lambda x: (x["NonComp"] * 100).round(1))
        .reset_index()
        .sort_values("Bookings", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    _top_routes["Pickup Region"] = _top_routes["Pickup Location"].map(_loc_map)
    _top_routes["Drop Region"] = _top_routes["Drop Location"].map(_loc_map)
    _top_routes = _top_routes[[
        "Pickup Location", "Pickup Region",
        "Drop Location", "Drop Region",
        "Bookings", "NonComp"
    ]]
    _top_routes.columns = [
        "Pickup", "Pickup Region", "Drop", "Drop Region",
        "Bookings", "Non-completion Rate (%)"
    ]

    with st.expander("📍 Top 10 most popular routes — with NCR region reference"):
        st.markdown(
            "The table below shows the **10 busiest routes** by booking volume, "
            "along with each location's NCR region. In Indian urban planning, "
            "**'Sector'** is a standard neighbourhood subdivision used in planned "
            "cities such as Noida, Gurgaon, and Faridabad (e.g. *Noida Sector 18* "
            "and *Noida Sector 62* are both in Noida but 10–15 km apart).\n\n"
            "**Note:** Distance and fare columns are excluded because cancelled "
            "bookings have no real trip data (the ride never started), which would "
            "skew the per-route averages."
        )
        st.dataframe(_top_routes, use_container_width=True, hide_index=True)

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
    fig4.colorbar(sm, ax=ax, label="Non-completion Rate (%)")
    ax.axis("off")
    st.pyplot(fig4)

    # Find the top hub nodes dynamically
    _top_hubs = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:3]
    _hub_names = ", ".join(_top_hubs)

    st.info(
        "**What this graph shows:** This is a **directed network graph** where each "
        "circle (node) represents a pickup or drop-off location in the NCR region. "
        "Arrows between nodes represent routes that passengers actually booked.\n\n"
        "**How to read it:**\n"
        "- **Line thickness** = how many bookings occurred on that route (thicker = "
        "more popular)\n"
        "- **Line colour** = non-completion rate on that route (**green** = low "
        "non-completion, good service; **yellow** = moderate; **red** = high "
        "non-completion, underserved)\n"
        "- **Circle size** = how many routes connect to that location (larger = "
        "more connected hub)\n\n"
        f"**Key trend:** The most connected hubs in the current view are "
        f"**{_hub_names}**. Routes radiating outward to less-connected peripheral "
        f"nodes tend to show warmer (redder) colours, indicating higher non-completion "
        f"rates on corridors leading away from the city centre. "
        f"Expand the **Top 10 most popular routes** table above to see which "
        f"NCR region each location belongs to (e.g. Delhi vs. Gurgaon / Noida)."
    )

    st.subheader("Top 10 Highest-Risk Routes")
    top_risk = (
        route_data[route_data["count"] > 1]
        .assign(cancel_rate=lambda x: (x["cancel_rate"] * 100).round(1))
        .sort_values("cancel_rate", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_risk.columns = ["Pickup", "Drop", "Total Bookings", "Non-completion Rate (%)"]
    st.dataframe(top_risk, use_container_width=True, hide_index=True)

    if len(top_risk) > 0:
        _worst_pickup = top_risk.iloc[0]["Pickup"]
        _worst_drop   = top_risk.iloc[0]["Drop"]
        _worst_rate   = top_risk.iloc[0]["Non-completion Rate (%)"]
        st.info(
            f"**What this table shows:** The 10 routes with the highest non-completion "
            f"rates (among routes with more than 1 booking). The worst-performing route "
            f"is **{_worst_pickup} \u2192 {_worst_drop}** at **{_worst_rate}%** non-completion. "
            f"Compare this to the dataset average of ~38%. Routes where either the "
            f"pickup or drop location is in a suburban region (see the "
            f"**Top 10 routes** table above for region classifications) tend to "
            f"cluster near the top of this list, suggesting a **mobility equity "
            f"gap** \u2014 passengers "
            f"in outer zones face systematically worse service because drivers are less "
            f"willing to accept longer trips with lower return-trip demand."
        )


# ── Tab 4: Model Insights ─────────────────────────────────────
with tab4:
    st.subheader("Trip Intervention Prediction Model — XGBoost")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model",         "XGBoost")
    col2.metric("CV Folds",      "5-fold")
    col3.metric("Test AUC",      "0.56")

    st.markdown("""
    ### What changed: cleaning error → model correction

    Our original model achieved an **AUC of 0.97** — but this was a **false signal**.
    The cleaning pipeline (Notebook 01) had applied unconditional median imputation
    to *all* rows, filling 48,000 cancelled bookings (which never started) with
    identical values: Distance = 23.72 km, Fare = ₹414. The model was simply
    detecting "which rows were imputed" — not learning real cancellation patterns.

    **After fixing the cleaning script** (restricting imputation to trips that
    actually started), we redesigned the model:

    - **Target:** `needs_intervention` = 1 when the trip was Incomplete (broke down
      mid-journey) OR Completed with either Driver or Customer Rating < 4.0
    - **Data:** 102,000 rows (Completed + Incomplete only — trips with real data)
    - **Features:** Only information available at trip start: Distance, Fare, Hour,
      Weekday, Month, Vehicle Type
    - **Ratings** are used to *define* the target (what counts as a bad outcome),
      NOT as input features — so there is no data leakage
    """)

    st.subheader("Feature Importance")
    fi = {
        "Ride Distance": 0.2505,
        "Booking Value (Passenger Fare)": 0.1606,
        "Hour": 0.1505,
        "Month": 0.1493,
        "Weekday": 0.1463,
        "Vehicle Type": 0.1427,
        "Is_Weekend": 0.0000,
    }
    fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
    fig5 = px.bar(
        fi_df.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        title="Feature Importance Scores (corrected model)",
        color="Importance", color_continuous_scale="Blues",
        template="plotly_white",
        labels={"Importance": "Importance Score"}
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.info(
        "**Reading this chart:** Unlike the original model where Ride Distance "
        "dominated at 92.7%, the corrected model shows **roughly even importance** "
        "across all features (~14–25% each). No single feature stands out as a "
        "strong predictor of poor trip outcomes. This means that whether a trip "
        "results in a breakdown or low rating is driven by factors **not captured "
        "in this dataset** — such as driver behaviour, vehicle condition, traffic, "
        "or weather."
    )

    st.markdown("""
    ### Key Findings

    - **AUC = 0.56** — the model performs only slightly better than random,
      confirming that trip-start features alone cannot reliably predict poor outcomes
    - **Feature importance is evenly distributed** — no single feature dominates,
      unlike the artifact-inflated original where Distance was 92.7%
    - **An anomalously high AUC (> 0.95) should always be investigated** — our
      original 0.97 turned out to be a data cleaning artifact, not real predictive power

    ### Lesson Learned
    This project demonstrates a critical real-world data science workflow:
    discovering a data pipeline error, diagnosing its downstream impact,
    correcting it, and **honestly reporting** the revised (lower) model
    performance. The corrected AUC of 0.56 is the truth — and recognising
    that truth is more valuable than reporting an inflated 0.97.

    ### Social Data Science Lens
    Underserved urban corridors show systematically higher non-completion rates,
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
        "Non-completion rate by vehicle type": """
SELECT   vehicle_type,
         COUNT(*)                           AS total_bookings,
         ROUND(AVG(is_cancelled)*100, 1)   AS non_completion_rate_pct
FROM     rides
GROUP BY vehicle_type
ORDER BY non_completion_rate_pct DESC""",

        "Top 10 highest-risk pickup locations": """
SELECT   pickup_location,
         COUNT(*)                           AS total_bookings,
         ROUND(AVG(is_cancelled)*100, 1)   AS non_completion_rate_pct
FROM     rides
GROUP BY pickup_location
HAVING   COUNT(*) > 100
ORDER BY non_completion_rate_pct DESC
LIMIT    10""",

        "Weekend vs Weekday comparison": """
SELECT   CASE WHEN is_weekend=1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
         COUNT(*)                           AS total_bookings,
         ROUND(AVG(booking_value), 1)       AS avg_passenger_fare,
         ROUND(AVG(ride_distance), 1)       AS avg_distance_km,
         ROUND(AVG(is_cancelled)*100, 1)   AS non_completion_rate_pct
FROM     rides
GROUP BY is_weekend""",

        "Payment method breakdown": """
SELECT   payment_method,
         COUNT(*)                           AS total_bookings,
         ROUND(AVG(is_cancelled)*100, 1)   AS non_completion_rate_pct,
         ROUND(AVG(booking_value), 1)       AS avg_passenger_fare
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




