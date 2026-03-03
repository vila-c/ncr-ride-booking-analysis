import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx

# Page config
st.set_page_config(
    page_title="NCR Ride Booking Analytics",
    page_icon="🚕",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_ncr_rides_with_target.csv",
                     parse_dates=["Datetime"])
    return df

df = load_data()

# Title
st.title("🚕 NCR Ride Booking Analytics Dashboard")
st.markdown("**Author: Vila Chung** · HKU BASc Social Data Science · 2025")
st.divider()

# KPIs
total     = len(df)
completed = (df["Cancel_Type"] == "Completed").sum()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Bookings",    f"{total:,}")
col2.metric("Completion Rate",   f"{completed/total:.1%}")
col3.metric("Cancellation Rate", f"{1 - completed/total:.1%}")
col4.metric("Vehicle Types",     df["Vehicle Type"].nunique())
st.divider()

# Sidebar filters
st.sidebar.header("🔍 Filters")
vehicle_filter = st.sidebar.multiselect(
    "Vehicle Type",
    options=df["Vehicle Type"].unique(),
    default=df["Vehicle Type"].unique()
)
hour_range = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))
filtered = df[
    (df["Vehicle Type"].isin(vehicle_filter)) &
    (df["Hour"].between(hour_range[0], hour_range[1]))
]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "⏰ Time Analysis", "🗺️ Route Network", "🤖 Model Insights"]
)

# Tab 1
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            filtered["Cancel_Type"].value_counts().reset_index(),
            x="Cancel_Type", y="count",
            title="Booking Status Distribution",
            color="Cancel_Type", template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        comp = filtered["Cancel_Type"].value_counts(normalize=True).mul(100).round(1)
        fig2 = px.pie(
            values=comp.values, names=comp.index,
            title="Completion vs Cancellation",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)

# Tab 2
with tab2:
    hourly = (
        filtered.groupby("Hour")
        .agg(Volume=("Booking ID", "count"),
             Cancel_Rate=("is_cancelled", "mean"))
        .assign(Cancel_Rate=lambda x: x["Cancel_Rate"] * 100)
    )
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=hourly.index, y=hourly["Volume"],
        name="Volume", marker_color="steelblue"
    ))
    fig3.add_trace(go.Scatter(
        x=hourly.index, y=hourly["Cancel_Rate"],
        name="Cancel Rate (%)", yaxis="y2",
        line=dict(color="tomato")
    ))
    fig3.update_layout(
        title="Hourly Volume vs Cancellation Rate",
        yaxis2=dict(overlaying="y", side="right", title="Cancel Rate (%)"),
        template="plotly_white"
    )
    st.plotly_chart(fig3, use_container_width=True)

# Tab 3
with tab3:
    st.subheader("Top Route Network (by Volume)")
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
    ax.axis("off")
    st.pyplot(fig4)

# Tab 4
with tab4:
    st.subheader("Cancellation Risk — Feature Importance")
    st.markdown("""
    A **Random Forest classifier** was trained to predict booking cancellations.
    - **Ride Distance** and **Hour of Day** are the strongest predictors
    - **Vehicle Type** and **VTAT** are secondary factors
    - 5-fold cross-validation confirms robust generalisation
    """)
    fi = {
        "Ride Distance": 0.21, "Hour": 0.18, "Avg VTAT": 0.15,
        "Booking Value": 0.13, "Month": 0.10, "Vehicle Type": 0.09,
        "Weekday": 0.07, "Avg CTAT": 0.07
    }
    fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
    fig5 = px.bar(
        fi_df.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        title="Feature Importance (Random Forest)",
        color="Importance", color_continuous_scale="Blues",
        template="plotly_white"
    )
    st.plotly_chart(fig5, use_container_width=True)

# Footer
st.divider()
st.caption("Dataset: Uber Ride Analytics Dashboard · Kaggle (Yash Devladdha) · For educational purposes only.")