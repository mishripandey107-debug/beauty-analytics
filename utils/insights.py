import numpy as np
import pandas as pd

from utils.segmentation import rfm_table


def generate_insights(df, rev):
    """Returns a list of (insight, recommended action)."""
    out = []
    total = df["revenue"].sum()

    # 1. Top category
    cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    if len(cat) and total:
        out.append((f"{cat.index[0]} drives {cat.iloc[0] / total * 100:.0f}% of revenue.",
                    "Protect stock and ad spend here, and use it to cross-sell other categories."))

    # 2. Biggest growth month
    m = df.groupby(pd.Grouper(key="order_date", freq="MS"))["revenue"].sum()
    g = m.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(g):
        out.append((f"{g.idxmax():%B %Y} had the biggest jump: revenue grew {g.max() * 100:.0f}% over the month before.",
                    "Look at what ran that month (festive push, discounts) and repeat it."))

    # 3. Revenue tied up in At Risk customers
    rfm = rfm_table(df)
    risk = rfm[rfm["segment"] == "At Risk"]
    if len(risk) and rfm["monetary"].sum():
        share = risk["monetary"].sum() / rfm["monetary"].sum() * 100
        out.append((f"{share:.0f}% of revenue came from {len(risk)} 'At Risk' customers who have gone quiet.",
                    "Send them a win-back offer. The list can be downloaded in the Customers tab."))

    # 4. Lowest rated product
    if len(rev):
        r = rev.groupby("name")["rating"].agg(["mean", "count"])
        r = r[r["count"] >= 10]
        if len(r):
            out.append((f"{r['mean'].idxmin()} has the lowest rating ({r['mean'].min():.1f}/5).",
                        "Read its negative reviews and fix quality or packaging issues first."))

    # 5. Category with weakest repeat rate
    cc = df.groupby(["category", "customer_id"])["order_id"].nunique().reset_index()
    rep = cc.groupby("category")["order_id"].apply(lambda s: (s > 1).mean() * 100)
    if len(rep):
        out.append((f"{rep.idxmin()} has the lowest repeat-purchase rate ({rep.min():.0f}%).",
                    "Try refill reminders, bundles, or a loyalty reward for this category."))
    return out