import numpy as np
import pandas as pd

SEGMENT_ORDER = ["Champions", "Loyal", "Potential Loyalists", "At Risk", "Lost"]


def _segment(r, f):
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Lost"
    return "Potential Loyalists"


def rfm_table(df):
    """One row per customer: Recency, Frequency, Monetary + segment."""
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    g = df.groupby("customer_id").agg(
        recency=("order_date", lambda x: (snapshot - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()
    # Recency: most recent buyers get the highest score (5)
    g["R"] = pd.cut(g["recency"].rank(method="first", pct=True),
                    [0, .2, .4, .6, .8, 1.0], labels=[5, 4, 3, 2, 1]).astype(int)
    # Frequency: 1 order = 1 ... 9+ orders = 5
    g["F"] = pd.cut(g["frequency"], [0, 1, 2, 4, 8, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
    g["segment"] = [_segment(r, f) for r, f in zip(g["R"], g["F"])]
    g["monetary"] = g["monetary"].round(0)
    return g


def cohort_retention(df, months=6):
    """% of each first-purchase-month group that bought again in months 1..6."""
    d = df[["customer_id", "order_date"]].copy()
    d["m"] = d["order_date"].dt.to_period("M")
    d["cohort"] = d.groupby("customer_id")["m"].transform("min")
    d["idx"] = (d["m"] - d["cohort"]).apply(lambda x: x.n)
    last = d["m"].max()
    t = d.groupby(["cohort", "idx"])["customer_id"].nunique().unstack(fill_value=0)
    t = t.reindex(columns=range(0, months + 1), fill_value=0)
    ret = t.iloc[:, 1:].div(t[0], axis=0) * 100
    for c in ret.index:
        avail = (last - c).n
        for k in ret.columns:
            if k > avail:
                ret.loc[c, k] = np.nan  # that month hasn't happened yet
    ret.index = ret.index.astype(str)
    ret.columns = [f"Month {k}" for k in ret.columns]
    return ret