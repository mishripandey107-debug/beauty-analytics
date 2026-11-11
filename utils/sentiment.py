import re
from collections import Counter

import pandas as pd
from textblob import TextBlob

STOP = set("""this that with have they them very from what were would your about also just
been into than then when which while there their these those some more most such only
really thing things""".split())


def add_sentiment(rev):
    rev = rev.copy()
    rev["polarity"] = rev["review_text"].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
    rev["sentiment"] = pd.cut(rev["polarity"], [-1.01, -0.05, 0.05, 1.01],
                              labels=["Negative", "Neutral", "Positive"])
    return rev


def complaint_table(rev, top=5):
    bad = rev[rev["rating"] <= 2]
    total = rev.groupby("name").size()
    rows = []
    for name, grp in bad.groupby("name"):
        skip = set(re.findall(r"[a-z]+", name.lower()))
        words = Counter(
            w for t in grp["review_text"]
            for w in re.findall(r"[a-z]+", str(t).lower())
            if len(w) > 3 and w not in STOP and w not in skip
        )
        rows.append({
            "product": name,
            "negative_reviews": len(grp),
            "negative_%": round(len(grp) / total[name] * 100, 1),
            "common_negative_words": ", ".join(w for w, _ in words.most_common(5)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("negative_reviews", ascending=False).head(top)