import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)  # fixed seed = same data every time
Path("data").mkdir(exist_ok=True)

START, END = pd.Timestamp("2024-11-01"), pd.Timestamp("2026-10-31")

# ---------- PRODUCTS ----------
catalog = {
    "Skincare": [("Vitamin C Serum", 699), ("Hyaluronic Acid Serum", 649), ("Niacinamide Serum", 599),
                 ("Daily Moisturizer", 449), ("Gel Face Wash", 299), ("Sunscreen SPF 50", 549),
                 ("Night Repair Cream", 899), ("Under Eye Cream", 499), ("Clay Face Mask", 399),
                 ("Lip Balm", 199)],
    "Makeup": [("Matte Lipstick", 399), ("Liquid Eyeliner", 249), ("Volumizing Mascara", 449),
               ("Compact Powder", 349), ("Foundation", 699), ("Kajal Pencil", 199),
               ("Blush Palette", 599), ("Lip Gloss", 299)],
    "Haircare": [("Onion Hair Oil", 349), ("Anti-Dandruff Shampoo", 399), ("Hair Serum", 499),
                 ("Argan Conditioner", 449), ("Hair Mask", 549), ("Keratin Shampoo", 599)],
    "Fragrance": [("Floral Perfume", 999), ("Body Mist", 499), ("Woody Eau de Parfum", 1299),
                  ("Roll-on Deodorant", 249), ("Gift Perfume Set", 1499), ("Travel Mist", 299)],
}
rows, n = [], 1
for cat, items in catalog.items():
    for name, price in items:
        rows.append({"product_id": f"P{n:03d}", "name": name, "category": cat, "price": price})
        n += 1
products = pd.DataFrame(rows)
price_map = dict(zip(products.product_id, products.price))
pop = rng.uniform(0.3, 1.5, len(products))  # how popular each product is
cat_groups = {c: products.index[products.category == c].to_numpy() for c in catalog}

def pick_product(fav):
    cat = fav if rng.random() < 0.7 else str(rng.choice(list(catalog)))
    idx = cat_groups[cat]
    p = pop[idx] / pop[idx].sum()
    return products.loc[rng.choice(idx, p=p), "product_id"]

# ---------- SEASONALITY ----------
days = pd.date_range(START, END)
season = {1: 0.9, 2: 1.15, 3: 0.9, 4: 0.85, 5: 0.9, 6: 0.85,
          7: 0.9, 8: 1.0, 9: 1.05, 10: 1.5, 11: 1.6, 12: 1.4}
valentine = np.array([1.5 if (d.month == 2 and 7 <= d.day <= 14) else 1.0 for d in days])
w = np.array([season[d.month] for d in days]) * valentine * np.linspace(0.7, 1.3, len(days))

# ---------- ORDERS ----------
cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
city_p = [0.22, 0.20, 0.18, 0.10, 0.09, 0.07, 0.08, 0.06]
N_CUSTOMERS = 2000
first_idx = rng.choice(len(days), size=N_CUSTOMERS, p=w / w.sum())

orders, oid = [], 1
for c in range(N_CUSTOMERS):
    cust_id = f"C{c + 1:04d}"
    city = str(rng.choice(cities, p=city_p))
    fav = str(rng.choice(list(catalog), p=[0.4, 0.3, 0.2, 0.1]))
    kind = rng.choice(["one_time", "occasional", "loyal"], p=[0.45, 0.35, 0.20])
    if kind == "one_time":
        n_orders = 1
    elif kind == "occasional":
        n_orders = 1 + rng.poisson(5)
    else:
        n_orders = 1 + rng.poisson(24)

    start_i = first_idx[c]
    remaining = len(days) - start_i
    n_orders = min(n_orders, 1 + remaining // 7)
    idxs = [start_i]
    if n_orders > 1 and remaining > 1:
        wr = w[start_i + 1:]
        idxs += list(rng.choice(np.arange(start_i + 1, len(days)), size=n_orders - 1, p=wr / wr.sum()))

    for i in sorted(idxs):
        d = days[i]
        pid = pick_product(fav)
        qty = int(rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1]))
        if d.month in (10, 11, 12):
            disc = int(rng.choice([0, 5, 10, 15, 20, 30], p=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1]))
        else:
            disc = int(rng.choice([0, 5, 10, 15, 20, 30], p=[0.5, 0.2, 0.15, 0.1, 0.05, 0.0]))
        price = price_map[pid]
        orders.append({
            "order_id": f"O{oid:06d}", "customer_id": cust_id, "order_date": d.date(),
            "product_id": pid, "quantity": qty, "price": price, "discount_pct": disc,
            "city": city, "revenue": round(qty * price * (1 - disc / 100), 2),
        })
        oid += 1

orders = pd.DataFrame(orders)

# ---------- REVIEWS ----------
problem = {"Foundation", "Sunscreen SPF 50", "Keratin Shampoo"}  # products with weaker reviews
pos = ["Love this {p}! Works really well and the quality is great.",
       "Amazing {p}, totally worth the price. Will buy again.",
       "My skin and hair feel so good since I started using the {p}.",
       "Best {p} I have tried. Fast delivery and lovely packaging.",
       "Great {p}, gives visible results. Highly recommend."]
neu = ["The {p} is okay, nothing special.",
       "Average {p}. Does the job but not amazing.",
       "Decent {p} for the price, packaging could be better."]
neg = ["Disappointed with the {p}. It caused breakouts and irritation.",
       "The {p} smells strange and the quality is poor.",
       "Waste of money, the {p} did not work at all.",
       "Terrible {p}, very sticky and the packaging leaked.",
       "The {p} is overpriced and the results are bad."]

counts = orders.product_id.value_counts().reindex(products.product_id).fillna(0)
rev_p = (counts / counts.sum()).to_numpy()
reviews = []
for r in range(4000):
    i = rng.choice(len(products), p=rev_p)
    name = products.loc[i, "name"]
    probs = [0.20, 0.20, 0.20, 0.25, 0.15] if name in problem else [0.05, 0.07, 0.13, 0.30, 0.45]
    rating = int(rng.choice([1, 2, 3, 4, 5], p=probs))
    pool = pos if rating >= 4 else neu if rating == 3 else neg
    text = str(rng.choice(pool)).format(p=name.lower())
    rdate = days[rng.integers(0, len(days))].date()
    reviews.append({"review_id": f"R{r + 1:05d}", "product_id": products.loc[i, "product_id"],
                    "rating": rating, "review_text": text, "date": rdate})
reviews = pd.DataFrame(reviews)

# ---------- SAVE ----------
orders.to_csv("data/orders.csv", index=False)
products.to_csv("data/products.csv", index=False)
reviews.to_csv("data/reviews.csv", index=False)
print(f"Done! {len(orders)} orders, {orders.customer_id.nunique()} customers, "
      f"{len(products)} products, {len(reviews)} reviews")
print(f"Total revenue: Rs {orders.revenue.sum():,.0f}")