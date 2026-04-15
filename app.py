import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    df = pd.read_csv("cleaned_ncr_rides_with_target.csv.gz",
                     parse_dates=["Datetime"])
    return df

df = load_data()

# Defensive fallback: ensure is_cancelled exists
if "is_cancelled" not in df.columns:
    df["is_cancelled"] = (df["Cancel_Type"] != "Completed").astype(int)

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
st.sidebar.markdown(
    "**Step 1.** Select **Vehicle Types** to focus on specific ride "
    "categories.\n\n"
    "**Step 2.** Drag the **Hour of Day** slider to narrow the time "
    "window.\n\n"
    "All five tabs update automatically."
)
st.sidebar.markdown("---")

vehicle_filter = st.sidebar.multiselect(
    "Vehicle Type",
    options=sorted(df["Vehicle Type"].unique()),
    default=sorted(df["Vehicle Type"].unique()),
    help="Select which vehicle types to include."
)
hour_range = st.sidebar.slider(
    "Hour of Day", 0, 23, (0, 23),
    help="Filter bookings by hour."
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
col3.metric("Non-completion Rate", f"{1 - completed/total:.1%}",
            help="Includes Driver Cancelled, Customer Cancelled, "
                 "No Driver Found, and Incomplete.")
col4.metric("Avg Passenger Fare",  f"₹{filtered['Booking Value'].mean():.0f}",
            help="Average fare for trips that actually started "
                 "(Completed + Incomplete).")
col5.metric("Avg Ride Distance",   f"{filtered['Ride Distance'].mean():.1f} km",
            help="Average distance for trips that actually started.")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "⏰ Time Analysis",
    "🗺️ Route Network",
    "🤖 Model Insights",
    "🗄️ SQL Explorer",
])

# ══════════════════════════════════════════════════════════════
# Tab 1: Overview
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Booking Status & Vehicle Distribution")

    with st.expander("⚠️ Data cleaning correction applied — click for details"):
        st.markdown(
            "The original cleaning pipeline applied **unconditional median "
            "imputation** to all rows, filling ~48,000 cancelled bookings "
            "(trips that never started) with identical placeholder values "
            "(Distance = 23.72 km, Fare = ₹414). The original XGBoost model "
            "learned to detect this imputation pattern (AUC 0.97), not real "
            "cancellation signals.\n\n"
            "**Fix applied:** Imputation restricted to Completed + Incomplete "
            "trips only. Corrected model AUC = ~0.56.\n\n"
            "**Lesson:** An anomalously high AUC (> 0.95) should always be "
            "investigated as a potential data leakage signal."
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

    # Compute dynamic stats for the insight box
    _comp_pct = (filtered["Cancel_Type"] == "Completed").mean() * 100
    _drv_pct = (filtered["Cancel_Type"] == "Driver Cancelled").mean() * 100
    _nodrv_pct = (filtered["Cancel_Type"] == "No Driver Found").mean() * 100
    _cust_pct = (filtered["Cancel_Type"] == "Customer Cancelled").mean() * 100

    st.info(
        f"**Key finding:** Only **{_comp_pct:.0f}%** of rides completed. "
        f"The biggest failure modes are **Driver Cancelled** ({_drv_pct:.0f}%) "
        f"and **No Driver Found** ({_nodrv_pct:.0f}%) — both supply-side "
        f"problems. Customer-initiated cancellations account for just "
        f"{_cust_pct:.0f}%."
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

    _vmin = vehicle_stats["Non-completion Rate (%)"].min()
    _vmax = vehicle_stats["Non-completion Rate (%)"].max()
    st.info(
        f"**Key finding:** All vehicle types have non-completion rates between "
        f"**{_vmin:.0f}%** and **{_vmax:.0f}%** — vehicle type has virtually "
        f"no impact on whether a ride gets completed."
    )


# ══════════════════════════════════════════════════════════════
# Tab 2: Time Analysis
# ══════════════════════════════════════════════════════════════
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

    _peak_hour = hourly["Volume"].idxmax()
    _peak_vol = hourly.loc[_peak_hour, "Volume"]
    _rate_range = hourly["Cancel_Rate"].max() - hourly["Cancel_Rate"].min()
    st.info(
        f"**Key finding:** Booking volume peaks at hour {_peak_hour} "
        f"(~{_peak_vol:,.0f} bookings), but the non-completion rate stays "
        f"flat across all 24 hours (range: just {_rate_range:.1f} pp). "
        f"**Time of day does not drive non-completion in this dataset.**"
    )

    with st.expander("📋 Hypothesis test: does peak hour have higher non-completion?"):
        st.markdown(
            "**Hypothesis:** Rush hours (7–9 AM, 17–20 PM) should show "
            "higher non-completion due to supply-demand imbalance."
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
            f"**Result:** Range across all time bins is just **{_range:.2f} pp**. "
            f"**Hypothesis not supported** — time of day has negligible impact."
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

    if len(day_stats) == 2:
        _wd_rate = day_stats.loc[day_stats["Day_Type"] == "Weekday", "Cancel_Rate"].values[0]
        _we_rate = day_stats.loc[day_stats["Day_Type"] == "Weekend", "Cancel_Rate"].values[0]
        st.info(
            f"**Key finding:** Weekday ({_wd_rate}%) vs Weekend ({_we_rate}%) "
            f"non-completion rates differ by just "
            f"**{abs(_wd_rate - _we_rate):.1f} pp** — not meaningful."
        )


# ══════════════════════════════════════════════════════════════
# Tab 3: Route Network — REDESIGNED
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Route Network & Mobility Equity Analysis")
    st.markdown(
        "This section maps the NCR ride network to reveal **which corridors "
        "are well-served** and **which are underserved**. Nodes represent "
        "pickup/drop locations; edges represent booked routes."
    )

    # ── Location classifier ──────────────────────────────────
    _all_locs = sorted(set(
        filtered["Pickup Location"].unique().tolist()
        + filtered["Drop Location"].unique().tolist()
    ))

    REGION_COLORS = {
        "Delhi":     "#4393c3",
        "Gurgaon":   "#2ca02c",
        "Noida":     "#ff7f0e",
        "Faridabad": "#9467bd",
        "Ghaziabad": "#d62728",
        "Outer NCR": "#8c564b",
    }

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

    # ── Compute route data ───────────────────────────────────
    all_route_data = (
        filtered.groupby(["Pickup Location", "Drop Location"])
        .agg(count=("is_cancelled", "size"),
             cancel_rate=("is_cancelled", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    # ── Section 1: Summary Metrics ───────────────────────────
    _total_corridors = len(all_route_data)
    _high_risk_routes = len(all_route_data[
        (all_route_data["cancel_rate"] > 0.45) &
        (all_route_data["count"] > 5)
    ])

    # Region-level non-completion rates
    _region_cancel = (
        filtered.assign(Region=filtered["Pickup Location"].map(_loc_map))
        .groupby("Region")["is_cancelled"]
        .agg(["mean", "count"])
        .query("count > 100")
        .sort_values("mean", ascending=False)
    )
    if len(_region_cancel) >= 2:
        _worst_region = _region_cancel.index[0]
        _worst_rate = _region_cancel["mean"].iloc[0] * 100
        _best_region = _region_cancel.index[-1]
        _best_rate = _region_cancel["mean"].iloc[-1] * 100
        _equity_gap = _worst_rate - _best_rate
    else:
        _worst_region, _best_region = "N/A", "N/A"
        _worst_rate, _best_rate, _equity_gap = 0, 0, 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Route Corridors", f"{_total_corridors:,}")
    m2.metric("High-Risk Routes (>45%)", f"{_high_risk_routes}")
    m3.metric("Equity Gap (worst vs best region)",
              f"{_equity_gap:.1f} pp",
              help=f"Difference between {_worst_region} ({_worst_rate:.1f}%) "
                   f"and {_best_region} ({_best_rate:.1f}%)")

    st.divider()

    # ── Section 2: Regional Overview (bar chart) ─────────────
    st.markdown("#### Non-completion Rate by Region")
    st.caption(
        "Each bar shows the non-completion rate for all bookings originating "
        "from that NCR region. Regions with fewer than 100 bookings are excluded."
    )

    _region_df = (
        _region_cancel.reset_index()
        .assign(Rate=lambda x: (x["mean"] * 100).round(1))
        .rename(columns={"Region": "NCR Region", "count": "Bookings"})
        .sort_values("Rate", ascending=True)
    )

    _avg_overall = filtered["is_cancelled"].mean() * 100

    fig_region = px.bar(
        _region_df, x="Rate", y="NCR Region", orientation="h",
        color="Rate",
        color_continuous_scale=["#2ca02c", "#fee08b", "#d73027"],
        range_color=[_region_df["Rate"].min() - 1, _region_df["Rate"].max() + 1],
        template="plotly_white",
        labels={"Rate": "Non-completion Rate (%)"},
        text="Rate",
        hover_data={"Bookings": ":,"},
    )
    fig_region.add_vline(
        x=_avg_overall, line_dash="dash", line_color="black",
        annotation_text=f"Avg: {_avg_overall:.1f}%",
        annotation_position="top right"
    )
    fig_region.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_region.update_layout(
        height=300,
        coloraxis_showscale=False,
        yaxis_title="",
        margin=dict(l=0, r=40, t=10, b=0),
    )
    st.plotly_chart(fig_region, use_container_width=True)

    if _equity_gap > 0:
        st.info(
            f"**Equity gap:** **{_worst_region}** has a non-completion rate of "
            f"**{_worst_rate:.1f}%**, which is **{_equity_gap:.1f} pp higher** "
            f"than **{_best_region}** ({_best_rate:.1f}%). Passengers in "
            f"peripheral zones face systematically worse service."
        )

    st.divider()

    # ── Section 3: Network Graph ─────────────────────────────
    st.markdown("#### Route Network Graph")

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        n_routes = st.slider(
            "Number of top routes to display", 10, 60, 30,
            help="Fewer routes = clearer graph. Start with 30."
        )
    with col_ctrl2:
        show_all_labels = st.checkbox("Show all labels", value=False,
                                      help="Display labels on every node")

    route_data = all_route_data.head(n_routes).copy()

    G = nx.DiGraph()
    for _, row in route_data.iterrows():
        G.add_edge(row["Pickup Location"], row["Drop Location"],
                   weight=row["count"],
                   cancel_rate=row["cancel_rate"] * 100)

    pos = nx.spring_layout(G, k=0.6, seed=42, iterations=60)
    edge_list   = list(G.edges(data=True))
    max_count   = route_data["count"].max()
    edge_widths = [d["weight"] / max_count * 6 + 0.5 for _, _, d in edge_list]
    edge_colors = [d["cancel_rate"] for _, _, d in edge_list]

    # Node properties
    node_regions = [_loc_map.get(n, "Delhi") for n in G.nodes()]
    node_colors  = [REGION_COLORS.get(r, "#999999") for r in node_regions]
    node_degrees = [G.degree(n) for n in G.nodes()]
    max_deg = max(node_degrees) if node_degrees else 1
    node_sizes   = [300 + (d / max_deg) * 1200 for d in node_degrees]

    # Determine which nodes get labels
    if show_all_labels:
        labels = {n: n for n in G.nodes()}
    else:
        # Show labels for: top hubs (degree > 2) or high-risk endpoints
        high_risk_nodes = set()
        for u, v, d in edge_list:
            if d["cancel_rate"] > 50:
                high_risk_nodes.add(u)
                high_risk_nodes.add(v)
        labels = {
            n: n for n in G.nodes()
            if G.degree(n) > 2 or n in high_risk_nodes
        }

    # Draw
    fig4, ax = plt.subplots(figsize=(14, 9))

    # Edges
    nx.draw_networkx_edges(G, pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in edge_list],
        width=edge_widths, edge_color=edge_colors,
        edge_cmap=plt.cm.RdYlGn_r, arrows=True,
        arrowsize=12, alpha=0.6,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=12, min_target_margin=12)

    # Nodes (colored by region)
    nx.draw_networkx_nodes(G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white", linewidths=1.5, alpha=0.9)

    # Labels
    nx.draw_networkx_labels(G, pos, labels, font_size=7,
                            font_weight="bold", ax=ax)

    # Colorbar for edges
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn_r,
        norm=plt.Normalize(
            vmin=min(edge_colors) if edge_colors else 0,
            vmax=max(edge_colors) if edge_colors else 100
        )
    )
    sm.set_array([])
    cbar = fig4.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Route Non-completion Rate (%)", fontsize=9)

    # Region legend
    legend_handles = [
        mpatches.Patch(color=c, label=r)
        for r, c in REGION_COLORS.items()
        if r in set(node_regions)
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title="NCR Region",
                  loc="lower left", fontsize=8, title_fontsize=9,
                  framealpha=0.9)

    ax.set_title(
        f"Top {n_routes} Routes — Node colour = region, "
        f"Edge colour = non-completion rate",
        fontsize=11, pad=12
    )
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig4)

    # Reading guide
    st.markdown(
        "**How to read this graph:**\n"
        "- **Node colour** = NCR region (see legend)\n"
        "- **Node size** = number of routes connected (larger = busier hub)\n"
        "- **Edge colour** = non-completion rate "
        "(🟢 green = good, 🟡 yellow = moderate, 🔴 red = underserved)\n"
        "- **Edge thickness** = booking volume (thicker = more popular)"
    )

    # Dynamic insight
    _top_hubs = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:3]
    _hub_regions = [_loc_map.get(h, "Delhi") for h in _top_hubs]
    _hub_text = ", ".join(
        [f"**{h}** ({r})" for h, r in zip(_top_hubs, _hub_regions)]
    )
    st.info(
        f"**Key hubs:** {_hub_text}. Routes extending to peripheral nodes "
        f"tend to show warmer (redder) colours, indicating higher "
        f"non-completion rates on corridors away from city centres."
    )

    st.divider()

    # ── Section 4: Top Routes Tables ─────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Busiest Routes")
        _top_routes = (
            all_route_data.head(10)
            .assign(cancel_rate=lambda x: (x["cancel_rate"] * 100).round(1))
            .reset_index(drop=True)
        )
        _top_routes["Pickup Region"] = _top_routes["Pickup Location"].map(_loc_map)
        _top_routes["Drop Region"] = _top_routes["Drop Location"].map(_loc_map)
        _display_top = _top_routes[[
            "Pickup Location", "Pickup Region",
            "Drop Location", "Drop Region",
            "count", "cancel_rate"
        ]].copy()
        _display_top.columns = [
            "Pickup", "Region", "Drop", "Region ",
            "Bookings", "Non-comp (%)"
        ]
        st.dataframe(_display_top, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("#### Highest-Risk Routes")
        top_risk = (
            all_route_data[all_route_data["count"] > 5]
            .assign(cancel_rate=lambda x: (x["cancel_rate"] * 100).round(1))
            .sort_values("cancel_rate", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        top_risk["Pickup Region"] = top_risk["Pickup Location"].map(_loc_map)
        top_risk["Drop Region"] = top_risk["Drop Location"].map(_loc_map)
        _display_risk = top_risk[[
            "Pickup Location", "Pickup Region",
            "Drop Location", "Drop Region",
            "count", "cancel_rate"
        ]].copy()
        _display_risk.columns = [
            "Pickup", "Region", "Drop", "Region ",
            "Bookings", "Non-comp (%)"
        ]
        st.dataframe(_display_risk, use_container_width=True, hide_index=True)

    if len(top_risk) > 0:
        _worst_pickup = top_risk.iloc[0]["Pickup Location"]
        _worst_drop   = top_risk.iloc[0]["Drop Location"]
        _worst_rate_r = top_risk.iloc[0]["cancel_rate"]
        _worst_p_region = _loc_map.get(_worst_pickup, "")
        _worst_d_region = _loc_map.get(_worst_drop, "")
        st.info(
            f"**Worst corridor:** {_worst_pickup} ({_worst_p_region}) → "
            f"{_worst_drop} ({_worst_d_region}) at **{_worst_rate_r}%** "
            f"non-completion (dataset avg: ~{_avg_overall:.0f}%). "
            f"Suburban and cross-region routes dominate the high-risk list, "
            f"suggesting a **mobility equity gap** — passengers in outer zones "
            f"face worse service due to lower driver supply."
        )

    with st.expander("💡 NCR Region Reference"):
        st.markdown(
            "In Indian urban planning, **'Sector'** is a standard neighbourhood "
            "subdivision in planned cities (e.g. *Noida Sector 18* and "
            "*Noida Sector 62* are both in Noida but 10–15 km apart).\n\n"
            "| Region | Key Areas |\n"
            "|--------|----------|\n"
            "| Delhi | Connaught Place, Saket, Dwarka, Rohini, etc. |\n"
            "| Gurgaon | DLF, Cyber Hub, Sikanderpur, Udyog Vihar |\n"
            "| Noida | Sectors 18/62/125, Greater Noida |\n"
            "| Ghaziabad | Indirapuram, Vaishali, Kaushambi |\n"
            "| Faridabad | Old/New Faridabad, NIT |\n"
            "| Outer NCR | Meerut, Sonipat, Panipat, Bhiwadi |"
        )


# ══════════════════════════════════════════════════════════════
# Tab 4: Model Insights
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Trip Intervention Prediction Model")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "XGBoost")
    col2.metric("CV Folds", "5-fold")
    col3.metric("Test AUC", "0.56")
    col4.metric("Avg Precision", "0.45")

    st.divider()

    # Before / After comparison
    st.markdown("#### What Changed: Data Cleaning Error → Model Correction")

    col_before, col_after = st.columns(2)
    with col_before:
        st.error("**Before (artifact)**")
        st.markdown(
            "- Target: `is_cancelled`\n"
            "- AUC: **0.97** (false signal)\n"
            "- Model detected imputation pattern, not real cancellation signals\n"
            "- 48K cancelled rows filled with identical placeholders "
            "(Dist=23.72 km, Fare=₹414)"
        )
    with col_after:
        st.success("**After (corrected)**")
        st.markdown(
            "- Target: `needs_intervention`\n"
            "- AUC: **0.56** (honest)\n"
            "- Imputation restricted to trips that actually started\n"
            "- Features: only trip-start info (Distance, Fare, Hour, "
            "Weekday, Month, Vehicle Type)\n"
            "- Ratings define target, not used as features (no leakage)"
        )

    st.divider()

    st.markdown("#### Feature Importance")
    fi = {
        "Ride Distance": 0.2505,
        "Booking Value": 0.1606,
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
        labels={"Importance": "Importance Score"},
        text="Importance",
    )
    fig5.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig5.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

    st.info(
        "**Key finding:** Feature importance is **evenly distributed** "
        "(~14–25% each). No single feature strongly predicts trip outcomes. "
        "Trip quality is driven by factors **not in this dataset** — driver "
        "behaviour, vehicle condition, traffic, weather."
    )

    st.divider()

    st.markdown("#### Takeaways")
    st.markdown(
        "1. **AUC > 0.95 is a red flag** — our original 0.97 was a data "
        "artifact, not real predictive power.\n"
        "2. **Trip-start features alone cannot predict outcomes** — the "
        "corrected AUC of 0.56 confirms limited predictive power.\n"
        "3. **Honesty over inflation** — reporting the corrected (lower) "
        "score is more valuable than keeping the inflated one."
    )

    with st.expander("🔬 Social Data Science Lens"):
        st.markdown(
            "Underserved urban corridors show systematically higher "
            "non-completion rates, raising **mobility equity** concerns "
            "for peripheral zones with limited alternative transport. "
            "Platform optimisation for profitability may unintentionally "
            "disadvantage communities in these areas."
        )


# ══════════════════════════════════════════════════════════════
# Tab 5: SQL Explorer
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("SQL Query Explorer")
    st.markdown(
        "Run SQL queries on the dataset using an in-memory SQLite database. "
        "Pick a preset or write your own query."
    )

    st.caption(
        "Table name: `rides` · Column names: snake_case "
        "(e.g. `vehicle_type`, `ride_distance`, `is_cancelled`)"
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
        "Choose a preset query:",
        list(PRESETS.keys()),
        help="Each preset answers a different business question."
    )
    query = st.text_area("SQL Query (editable):", value=PRESETS[preset], height=160)

    if st.button("▶ Run Query"):
        try:
            result = pd.read_sql(query, conn)
            st.success(f"Query returned {len(result)} rows.")
            st.dataframe(result, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"SQL Error: {e}")
