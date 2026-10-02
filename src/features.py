import pandas as pd

CUTOFF_START = "2010-06-01"
CUTOFF_END = "2011-11-01"

def build_snapshot(orders, returns, cutoff, window=30):
    cutoff = pd.Timestamp(cutoff)
    hist = orders[orders["order_date"] < cutoff].sort_values(["customer_id", "order_date"]).copy()
    hist["gap"] = hist.groupby("customer_id")["order_date"].diff().dt.days
    
    feat = hist.groupby("customer_id").agg(
        frequency=("order_date", "count"),
        monetary_total=("revenue", "sum"),
        aov=("revenue", "mean"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
        avg_items=("n_items", "mean"),
        avg_products=("n_products", "mean"),
        gap_mean=("gap", "mean"),
        gap_std=("gap", "std"),
        gap_last=("gap", "last"),
        country=("country", "last")
    )
    feat["recency"] = (cutoff - feat["last_order"]).dt.days
    feat["tenure"] = (cutoff - feat["first_order"]).dt.days
    feat["one_order"] = (feat["frequency"] == 1).astype(int)
    feat["recency_vs_gap"] = feat["recency"] / (feat["gap_mean"] + 1)
    feat["is_uk"] = (feat["country"] == "United Kingdom").astype(int)
    
    for d in [30, 90]:
        recent = hist[hist["order_date"] >= cutoff - pd.Timedelta(days=d)]
        feat[f"orders_{d}d"] = recent.groupby("customer_id")["order_date"].count()
    feat[["orders_30d", "orders_90d"]] = feat[["orders_30d", "orders_90d"]].fillna(0)
    feat["order_ratio_30_90"] = feat["orders_30d"] / (feat["orders_90d"] + 1)
    
    ret = returns[returns["order_date"] < cutoff]
    feat["return_count"] = ret.groupby("customer_id")["invoice"].nunique()
    feat["return_count"] = feat["return_count"].fillna(0)
    feat["return_ratio"] = feat["return_count"] / feat["frequency"]
    feat["cutoff_month"] = cutoff.month
    
    end = cutoff + pd.Timedelta(days=window)
    future = orders[(orders["order_date"] >= cutoff) & (orders["order_date"] < end)]
    feat["target"] = feat.index.isin(future["customer_id"].unique()).astype(int)
    feat["cutoff"] = cutoff
    return feat.drop(columns=["first_order", "last_order", "country"]).reset_index()

def build_dataset(orders, returns, window=30):
    cutoffs = pd.date_range(CUTOFF_START, CUTOFF_END, freq="MS")
    return pd.concat([build_snapshot(orders, returns, c, window) for c in cutoffs],
                    ignore_index=True)