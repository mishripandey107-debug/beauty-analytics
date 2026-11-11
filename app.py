import pandas as pd
import plotly.express as px
import streamlit as st

from utils.metrics import (prepare_orders, compute_kpis, mom_deltas,
                           monthly_revenue, product_table, fmt_inr)
from utils.segmentation import rfm_table, cohort_retention, SEGMENT_ORDER
from utils.sentiment import add_sentiment, complaint_table
from utils.insights import generate_insights
st.set_page_config(page_title="Beauty Growth Analytics", page_icon="💄", layout="wide")

PALETTE = ["#D6336C", "#F783AC", "#9C36B5", "#FFA8A8", "#7048E8", "#FFD8A8"]


@st.cache_data
def load_default():
    products = pd.read_csv("data/products.csv")
    reviews = pd.read_csv("data/reviews.csv", parse_dates=["date"])
    orders = prepare_orders(pd.read_csv("data/orders.csv"), products)
    return orders, products, reviews


orders, products, reviews = load_default()

# ---------------- SIDEBAR ----------------
st.sidebar.title("💄 Filters")
uploaded = st.sidebar.file_uploader("Upload your own orders CSV", type="csv")
if uploaded:
    try:
        orders = prepare_orders(pd.read_csv(uploaded), products)
        st.sidebar.success("Using your uploaded data")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
else:
    st.sidebar.caption("Showing sample data")

min_d, max_d = orders["order_date"].min().date(), orders["order_date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d),
                                   min_value=min_d, max_value=max_d)
if len(date_range) != 2:
    st.info("Please select both a start and an end date.")
    st.stop()

cats = sorted(orders["category"].unique())
cities = sorted(orders["city"].unique())
sel_cats = st.sidebar.multiselect("Category", cats, default=cats)
sel_cities = st.sidebar.multiselect("City", cities, default=cities)

df = orders[
    (orders["order_date"] >= pd.Timestamp(date_range[0]))
    & (orders["order_date"] <= pd.Timestamp(date_range[1]))
    & (orders["category"].isin(sel_cats))
    & (orders["city"].isin(sel_cities))
]
if df.empty:
    st.warning("No data for these filters. Try widening them.")
    st.stop()

# ---------------- HEADER + KPIs ----------------
st.title("💄 Beauty Brand Growth Analytics")
st.caption("Sample data for demonstration. Upload your own CSV in the sidebar.")

k = compute_kpis(df)
d = mom_deltas(df)
c1, c2, c3, c4 = st.columns(4)
fmt = lambda v: None if v is None else f"{v:+.1f}% vs last month"
c1.metric("Total Revenue", fmt_inr(k["revenue"]), fmt(d["revenue"]))
c2.metric("Orders", f"{k['orders']:,}", fmt(d["orders"]))
c3.metric("Avg Order Value", f"₹{k['aov']:,.0f}", fmt(d["aov"]))
c4.metric("Repeat Customers", f"{k['repeat']:.1f}%")

st.divider()
# reviews filtered by the same sidebar choices
rev = reviews.merge(products[["product_id", "name", "category"]], on="product_id", how="left")
rev = rev[rev["category"].isin(sel_cats)
          & (rev["date"] >= pd.Timestamp(date_range[0]))
          & (rev["date"] <= pd.Timestamp(date_range[1]))]
# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Products", "Customers", "Reviews"])

with tab1:
    st.subheader("💡 Key Insights")
    for text, action in generate_insights(df, rev):
        st.success(f"**{text}**\n\n👉 {action}")
    st.divider()

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Monthly revenue")
        m = monthly_revenue(df)
        fig = px.line(m, x="month", y="revenue", markers=True,
                      color_discrete_sequence=PALETTE)
        fig.update_layout(xaxis_title="", yaxis_title="Revenue (₹)", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Revenue by category")
        cat = df.groupby("category")["revenue"].sum().reset_index()
        fig = px.pie(cat, names="category", values="revenue", hole=0.55,
                     color_discrete_sequence=PALETTE)
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    pt = product_table(df, reviews)
    st.subheader("Top 10 products by revenue")
    top = pt.head(10).sort_values("revenue")
    fig = px.bar(top, x="revenue", y="name", orientation="h", color="category",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(xaxis_title="Revenue (₹)", yaxis_title="", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price vs units sold")
    fig = px.scatter(pt, x="price", y="units", size="revenue", color="category",
                     hover_name="name", color_discrete_sequence=PALETTE, size_max=45)
    fig.update_layout(xaxis_title="Price (₹)", yaxis_title="Units sold")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All products")
    st.dataframe(pt.drop(columns=["product_id"]), use_container_width=True, hide_index=True)

with tab3:
    rfm = rfm_table(df)
    summ = (rfm.groupby("segment")
               .agg(customers=("customer_id", "count"), revenue=("monetary", "sum"))
               .reindex(SEGMENT_ORDER).fillna(0).reset_index())

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Customers per segment")
        fig = px.bar(summ, x="segment", y="customers", color="segment",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.subheader("Revenue share by segment")
        fig = px.pie(summ, names="segment", values="revenue", hole=0.55,
                     color_discrete_sequence=PALETTE)
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cohort retention")
    st.caption("Of the customers who first bought in a given month, what % came back in each later month?")
    ret = cohort_retention(df)
    if ret.empty:
        st.info("Not enough data for cohorts.")
    else:
        fig = px.imshow(ret, text_auto=".0f", color_continuous_scale="RdPu", aspect="auto",
                        labels=dict(x="", y="First purchase month", color="% active"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("At Risk customers")
    risk = rfm[rfm["segment"] == "At Risk"].sort_values("monetary", ascending=False)
    st.dataframe(risk, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download At Risk customers (CSV)", risk.to_csv(index=False),
                       "at_risk_customers.csv", "text/csv")

with tab4:
    if rev.empty:
        st.info("No reviews for these filters.")
    else:
        rs = add_sentiment(rev)
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Average rating per product")
            avg = rs.groupby("name")["rating"].mean().round(2).sort_values().reset_index()
            fig = px.bar(avg, x="rating", y="name", orientation="h",
                         color_discrete_sequence=PALETTE, range_x=[1, 5])
            fig.update_layout(height=650, xaxis_title="Avg rating", yaxis_title="", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            st.subheader("Sentiment distribution")
            dist = rs["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).reset_index()
            dist.columns = ["sentiment", "reviews"]
            fig = px.bar(dist, x="sentiment", y="reviews", color="sentiment",
                         color_discrete_map={"Positive": "#2F9E44", "Neutral": "#ADB5BD", "Negative": "#D6336C"})
            fig.update_layout(showlegend=False, xaxis_title="", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 5 most complained-about products")
        st.dataframe(complaint_table(rs), use_container_width=True, hide_index=True)