"""
rfm_analysis.py

Scores every customer on Recency, Frequency and Monetary value, then assigns a
segment label.

SCORING
-------
Recency and monetary use tie-aware quintiles: customers with identical values
always receive identical scores, even when that makes the buckets uneven.

Frequency uses explicit order-count bands instead of quintiles, because its
distribution is too concentrated for quintiles to mean anything -- about 72% of
customers place exactly one order.

An earlier version applied `rank(method="first")` before `pd.qcut` to force five
equal-sized buckets. That stopped `qcut` raising a duplicate-bin-edge error, but
it broke ties by row order, so customers with identical purchase counts received
different scores: the one-order customers were spread across F-scores 1 through
4, which put 18 one-time buyers in Champions and 109 in Loyal Customers. Passing
`duplicates="drop"` to `qcut` is not an alternative -- it raises a different
error when explicit labels are supplied. Bands keep tied customers together and
make the thresholds visible.

This module is imported by scenario_simulator.py so the scoring exists in one
place. It used to be copied there, which is how two copies drift apart the
moment one of them is corrected.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SALES_DATA_PATH = os.path.join(DATA_DIR, "raw_sales_data.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "rfm_segments.csv")

# (minimum orders, maximum orders) per score, ascending. None = no upper bound.
FREQUENCY_BANDS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, None))

QUINTILES = 5


def score_tie_aware(values: pd.Series, ascending: bool = True,
                    buckets: int = QUINTILES) -> pd.Series:
    """Quintile scores in which tied values always share a score.

    Percentile ranking with `method="average"` gives every tied value the same
    percentile, so they land in the same bucket. Buckets end up uneven whenever
    values repeat, which is the correct trade: two customers who behaved
    identically must not be scored differently.
    """
    pct = values.rank(pct=True, method="average", ascending=ascending)
    return np.ceil(pct * buckets).clip(1, buckets).astype(int)


def score_frequency(frequency: pd.Series) -> pd.Series:
    """Score purchase counts against explicit bands (see FREQUENCY_BANDS)."""
    scores = pd.Series(1, index=frequency.index, dtype=int)
    for score, (low, high) in enumerate(FREQUENCY_BANDS, start=1):
        in_band = frequency >= low if high is None else frequency.between(low, high)
        scores[in_band] = score
    return scores


def assign_segment(row) -> str:
    """Assign a segment from the three scores.

    Frequency decides first, because it is the scarce signal in this customer
    base: about 77% of customers place exactly one order. Rules that test
    recency first (the classic RFM grid) collapse 91% of customers into two
    labels here and leave "About to Sleep / Average" with nobody in it.

    So repeat buyers -- f_score >= 3, meaning three or more orders -- are split
    by recency and value, and everyone else is split by recency alone.
    """
    r, f, m = row["r_score"], row["f_score"], row["m_score"]
    if f >= 3:
        if r >= 4 and m >= 4:
            return "Champions"
        if r >= 3:
            return "Loyal Customers"
        return "At Risk / Can't Lose Them"
    if r >= 4:
        return "Promising / New"
    if r == 3:
        return "About to Sleep / Average"
    return "Hibernating / Lost"


def compute_rfm(sales: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """Per-customer RFM values, scores and segment, indexed by CustomerID.

    `snapshot_date` defaults to one day after the last transaction in `sales`,
    so it stays correct if the simulator's date window changes. Pass it
    explicitly when scoring a historical slice.
    """
    df = sales.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalCost"] = df["UnitPrice"] * df["Quantity"]
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    grouped = df.groupby("CustomerID")
    rfm = pd.DataFrame({
        "recency": (snapshot_date - grouped["InvoiceDate"].max()).dt.days,
        "frequency": grouped["InvoiceNo"].nunique(),
        "monetary": grouped["TotalCost"].sum(),
    })

    # Recency is inverted: fewer days since the last order is better.
    rfm["r_score"] = score_tie_aware(rfm["recency"], ascending=False)
    rfm["f_score"] = score_frequency(rfm["frequency"])
    rfm["m_score"] = score_tie_aware(rfm["monetary"], ascending=True)
    rfm["customer_segment"] = rfm.apply(assign_segment, axis=1)
    return rfm


def describe(rfm: pd.DataFrame) -> None:
    print(f"Scored {len(rfm):,} customers.")
    print("\nfrequency score bands (tied customers share a score):")
    bands = rfm.groupby("f_score").agg(
        orders_min=("frequency", "min"), orders_max=("frequency", "max"),
        customers=("frequency", "size"))
    print(bands.to_string())
    print("\nsegments:")
    seg = rfm.groupby("customer_segment").agg(
        customers=("monetary", "size"), revenue=("monetary", "sum"))
    seg["cust_%"] = (seg["customers"] / seg["customers"].sum() * 100).round(1)
    seg["rev_%"] = (seg["revenue"] / seg["revenue"].sum() * 100).round(1)
    print(seg.sort_values("revenue", ascending=False).to_string())

    # A one-order customer in a "bought frequently" segment means the tie
    # handling has regressed; it was the original symptom of the qcut bug.
    frequent = {"Champions", "Loyal Customers", "At Risk / Can't Lose Them"}
    bad = rfm[(rfm["frequency"] == 1) & rfm["customer_segment"].isin(frequent)]
    if len(bad):
        print(f"\n[Warning] {len(bad)} one-order customers landed in a segment "
              f"that implies repeat buying. Check the scoring.")
    else:
        print("\nNo one-order customer is in a repeat-buyer segment, as expected.")


def main() -> None:
    print("Executing RFM pipeline...")
    sales = pd.read_csv(SALES_DATA_PATH)
    rfm = compute_rfm(sales)
    rfm.reset_index().to_csv(OUTPUT_PATH, index=False)
    describe(rfm)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
