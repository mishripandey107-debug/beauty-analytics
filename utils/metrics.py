import pandas as pd

REQUIRED = ["order_id", "customer_id", "order_date", "product_id",
            "quantity", "price", "discount_pct", "city"]


def prepare_orders(df, products):
    """Cleans an orders table and adds product name and category."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    if "revenue" not in df.columns:
        df["revenue"] = df["quantity"] * df["price"] * (1 - df["discount_pct"] / 100)
    df = df.drop(columns=[c for c in ["name", "category"] if c in df.columns])
    df = df.merge(products[["product_id", "name", "category"]], on="product_id", how="left")
    df["category"] = df["category"].fillna("Other")
    df["name"] = df["name"].fillna(df["product_id"])
    return df


def compute_kpis(df):
    total_rev = df["revenue"].sum()
    n_orders = df["order_id"].nunique()
    aov = total_rev / n_orders if n_orders else 0
    per_customer = df.groupby("customer_id")["order_id"].nunique()
    repeat_pct = (per_customer > 1).mean() * 100 if len(per_customer) else 0
    return {"revenue": total_rev, "orders": n_orders, "aov": aov, "repeat": repeat_pct}


def mom_deltas(df):
    """% change of the latest month vs the month before it."""
    m = df.groupby(pd.Grouper(key="order_date", freq="MS")).agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique"))
    m["aov"] = m["revenue"] / m["orders"].replace(0, pd.NA)
    out = {"revenue": None, "orders": None, "aov": None}
    if len(m) >= 2:
        last, prev = m.iloc[-1], m.iloc[-2]
        for k in out:
            if pd.notna(prev[k]) and prev[k] != 0 and pd.notna(last[k]):
                out[k] = (last[k] - prev[k]) / prev[k] * 100
    return out


def monthly_revenue(df):
    m = df.groupby(pd.Grouper(key="order_date", freq="MS"))["revenue"].sum().reset_index()
    m.columns = ["month", "revenue"]
    return m


def product_table(df, reviews):
    t = df.groupby(["product_id", "name", "category"]).agg(
        revenue=("revenue", "sum"), units=("quantity", "sum"), price=("price", "first")
    ).reset_index()
    avg = reviews.groupby("product_id")["rating"].mean().round(2).rename("avg_rating")
    t = t.merge(avg, on="product_id", how="left")
    t["revenue"] = t["revenue"].round(0)
    return t.sort_values("revenue", ascending=False)


def fmt_inr(x):
    if x >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"