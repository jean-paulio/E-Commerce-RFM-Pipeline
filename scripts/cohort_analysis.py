"""
cohort_analysis.py

Cohort retention analysis: groups customers by the month they were acquired,
then tracks what share of each cohort is still buying in the months after.

Answers a question RFM cannot: not "who is valuable now" but "is the business
acquiring better customers over time, and how fast do they decay". A cohort
table separates those two things, because every cohort is measured from its
own month zero rather than from the calendar.

READING THE TRIANGLE
--------------------
The matrix is triangular because later cohorts have had less time to be
observed. That right-censoring is the main trap in cohort analysis: averaging
down a column mixes different sets of cohorts, so a "month +8" average built
from four early cohorts is not comparable to a "month +1" average built from
eleven. Column sample sizes are printed alongside the averages, and cohort
quality is compared at a FIXED offset -- which is apples-to-apples -- rather
than by comparing each cohort's longest available run.

Small cohorts are the other trap. A cohort of three customers produces cells
reading 33% or 67% that look like signal on a heatmap and are one person's
coin flip. Cohorts below MIN_COHORT_SIZE are flagged and excluded from the
summary statistics.

Outputs:
    data/cohort_retention.csv       wide matrix (rows = cohort)
    data/cohort_retention_long.csv  tidy long format for Power BI / BI tools
    reports/charts/cohort_retention.png
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CHART_DIR = os.path.join(SCRIPT_DIR, "..", "reports", "charts")

SALES_FILE = os.path.join(DATA_DIR, "raw_sales_data.csv")
WIDE_FILE = os.path.join(DATA_DIR, "cohort_retention.csv")
LONG_FILE = os.path.join(DATA_DIR, "cohort_retention_long.csv")

# Below this, a cohort's retention percentages are too noisy to read as signal.
MIN_COHORT_SIZE = 30

NAVY = "#1F3864"
GOLD = "#C9973E"
BODY_GRAY = "#444444"


# --------------------------------------------------------------------------
# data prep
# --------------------------------------------------------------------------

def load_sales() -> pd.DataFrame:
    if not os.path.exists(SALES_FILE):
        print(f"Error: {SALES_FILE} not found. Run simulator.py first.")
        sys.exit(1)
    sales = pd.read_csv(SALES_FILE, parse_dates=["InvoiceDate"])
    sales["Revenue"] = sales["UnitPrice"] * sales["Quantity"]
    return sales


def drop_partial_tail(sales: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Drop a trailing month that holds only a few days of data.

    A partial final month shows up as a false collapse in the newest cohort's
    retention. Detected from the data rather than hardcoded, so it stays
    correct if simulator.py's window moves.
    """
    month = sales["InvoiceDate"].dt.to_period("M")
    last = month.max()
    days = sales.loc[month == last, "InvoiceDate"].dt.day.nunique()
    if days <= 5 and month.nunique() > 1:
        return sales[month < last].copy(), last
    return sales, None


def build_cohorts(sales: pd.DataFrame) -> pd.DataFrame:
    """Tag each transaction with its customer's cohort and months-since."""
    sales = sales.copy()
    sales["month"] = sales["InvoiceDate"].dt.to_period("M")
    first = sales.groupby("CustomerID")["month"].min().rename("cohort")
    sales = sales.join(first, on="CustomerID")
    sales["offset"] = (sales["month"] - sales["cohort"]).apply(lambda x: x.n)
    return sales


# --------------------------------------------------------------------------
# matrices
# --------------------------------------------------------------------------

def observed_mask(index, columns, last_period) -> pd.DataFrame:
    """True where a (cohort, offset) cell falls inside the observation window.

    Distinguishes "not yet observed" from "observed, nobody bought". groupby
    yields no row in either case, so without this a month where a cohort went
    quiet renders blank exactly like a month that has not happened yet -- and
    the reader cannot tell a dead cohort from a young one.
    """
    return pd.DataFrame(
        {c: [(last_period - cohort).n >= c for cohort in index] for c in columns},
        index=index,
    )


def retention_matrix(sales: pd.DataFrame, last_period):
    sizes = sales.groupby("cohort")["CustomerID"].nunique()
    active = (sales.groupby(["cohort", "offset"])["CustomerID"]
              .nunique().unstack())
    full_cols = range(0, int(active.columns.max()) + 1)
    active = active.reindex(columns=full_cols)

    mask = observed_mask(active.index, full_cols, last_period)
    active = active.where(~(mask & active.isna()), 0)   # observed but zero -> 0
    active = active.where(mask)                          # unobserved -> NaN

    retention = active.div(sizes, axis=0) * 100
    return retention, active, sizes, mask


def revenue_matrix(sales: pd.DataFrame, sizes: pd.Series, mask: pd.DataFrame):
    revenue = (sales.groupby(["cohort", "offset"])["Revenue"].sum()
               .unstack().reindex(columns=mask.columns))
    revenue = revenue.where(~(mask & revenue.isna()), 0).where(mask)
    per_member = revenue.div(sizes, axis=0)

    # Cumulative across observed cells only, so a censored tail stays NaN
    # rather than flattening into a plateau that reads as saturation.
    cumulative = per_member.copy()
    for row in cumulative.index:
        seen = cumulative.loc[row].notna()
        cumulative.loc[row, seen] = cumulative.loc[row, seen].cumsum()
    return per_member, cumulative


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def header(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def report_sizes(sizes, excluded_period):
    header("COHORT SIZES")
    if excluded_period is not None:
        print(f"  ({excluded_period} excluded -- partial month)\n")
    small = sizes[sizes < MIN_COHORT_SIZE]
    for period, n in sizes.items():
        flag = "  <- too small to read as signal" if n < MIN_COHORT_SIZE else ""
        print(f"  {period}   {n:>5,}{flag}")
    total = sizes.sum()
    print(f"\n  {len(sizes)} cohorts, {total:,} customers, "
          f"{sizes.min()}-{sizes.max()} per cohort")
    if len(small):
        print(f"  {len(small)} cohort(s) below the {MIN_COHORT_SIZE}-customer "
              f"threshold are excluded from averages.")
    else:
        print(f"  All cohorts are at or above the {MIN_COHORT_SIZE}-customer "
              f"threshold.")
    return sizes[sizes >= MIN_COHORT_SIZE].index


def report_matrix(retention, sizes, title, fmt="{:.0f}"):
    header(title)
    cols = [c for c in retention.columns if c <= 11]
    head = "  cohort      n  " + "".join(f"{('+' + str(c)):>6}" for c in cols)
    print(head)
    for period in retention.index:
        cells = ""
        for c in cols:
            v = retention.loc[period, c]
            cells += f"{'':>6}" if pd.isna(v) else f"{fmt.format(v):>6}"
        print(f"  {period} {sizes[period]:>6,}  {cells}")


def report_average_curve(retention, valid_index):
    header("AVERAGE RETENTION CURVE (censoring made explicit)")
    print("  Each offset averages only the cohorts old enough to have reached")
    print("  it, so the sample changes as you move right. Compare the counts.\n")
    print(f"  {'offset':<10}{'retention':>11}{'cohorts':>10}")
    sub = retention.loc[valid_index]
    for c in sorted(sub.columns):
        if c > 11:
            continue
        col = sub[c].dropna()
        if col.empty:
            continue
        label = "month 0" if c == 0 else f"month +{c}"
        print(f"  {label:<10}{col.mean():>10.1f}%{len(col):>10}")


def report_cohort_quality(retention, valid_index, offset=1):
    """Compare cohorts at a fixed offset -- the only fair comparison."""
    header(f"COHORT QUALITY AT A FIXED OFFSET (month +{offset})")
    print("  Comparing every cohort at the same age avoids the censoring bias.")
    print("  A trend here would mean acquisition quality is genuinely shifting.\n")
    if offset not in retention.columns:
        print("  offset not present in data")
        return
    series = retention.loc[valid_index, offset].dropna()
    if len(series) < 3:
        print("  too few cohorts to assess a trend")
        return

    for period, val in series.items():
        bar = "#" * int(round(val))
        print(f"  {period}  {val:>5.1f}%  {bar}")

    # A slope is not a trend. With ~11 noisy points a visually convincing
    # gradient appears easily by chance, so test whether the slope is
    # distinguishable from zero rather than just comparing it to the spread.
    from scipy import stats

    x = np.arange(len(series))
    fit = stats.linregress(x, series.values)
    spread = series.max() - series.min()
    print(f"\n  mean {series.mean():.1f}%, spread {spread:.1f} pp, "
          f"sd {series.std():.1f} pp")
    print(f"  linear trend: {fit.slope:+.2f} pp per cohort "
          f"(SE {fit.stderr:.2f}, p = {fit.pvalue:.3f}, n = {len(series)})")

    if fit.pvalue >= 0.05:
        print("  -> NOT significant at p < 0.05. Cohort quality is flat; the")
        print("     apparent gradient is sampling noise. Reading a trend into")
        print("     this would be the classic cohort-analysis error.")
    else:
        direction = "improving" if fit.slope > 0 else "declining"
        print(f"  -> significant at p < 0.05: acquisition quality is {direction}")
        print(f"     by roughly {abs(fit.slope):.2f} pp per cohort.")


def report_revenue(cumulative, sizes, valid_index):
    header("CUMULATIVE REVENUE PER COHORT MEMBER")
    print("  Dollars delivered per acquired customer, by cohort age. This is")
    print("  the payback view -- retention percentages do not show value.\n")
    sub = cumulative.loc[valid_index]
    cols = [c for c in sub.columns if c <= 11]
    print("  cohort      n  " + "".join(f"{('+' + str(c)):>8}" for c in cols))
    for period in sub.index:
        cells = ""
        for c in cols:
            v = sub.loc[period, c]
            cells += f"{'':>8}" if pd.isna(v) else f"{('$' + format(v, ',.0f')):>8}"
        print(f"  {period} {sizes[period]:>6,}  {cells}")

    first_month = sub[0].dropna()
    if 3 in sub.columns:
        by_m3 = sub[3].dropna()
        if len(by_m3):
            uplift = by_m3.mean() / first_month.loc[by_m3.index].mean() - 1
            print(f"\n  By month +3 a cohort has delivered "
                  f"{uplift*100:.0f}% more per member than at acquisition.")


# --------------------------------------------------------------------------
# chart
# --------------------------------------------------------------------------

def save_heatmap(retention, sizes, valid_index):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    os.makedirs(CHART_DIR, exist_ok=True)
    sub = retention.loc[valid_index]
    cols = [c for c in sub.columns if c <= 11]
    vals = sub[cols].values

    # Month 0 is 100% by construction and would dominate any colour scale,
    # so scale to the post-acquisition cells only.
    post = vals[:, 1:]
    vmax = np.nanmax(post) if np.isfinite(np.nanmax(post)) else 100

    cmap = LinearSegmentedColormap.from_list("navy", ["#FFFFFF", "#AFC0DC", NAVY])
    cmap.set_bad("#F5F5F5")

    fig, ax = plt.subplots(figsize=(11, 5.6))
    masked = np.ma.masked_invalid(vals)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(cols)),
                  ["Month 0" if c == 0 else f"+{c}" for c in cols], fontsize=9)
    ax.set_yticks(range(len(sub.index)),
                  [f"{p}  (n={sizes[p]:,})" for p in sub.index], fontsize=9)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            shade = 0 if j == 0 else v / vmax if vmax else 0
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                    color="white" if (j == 0 or shade > 0.55) else "#222222")

    ax.set_title("Cohort retention — share of each acquisition cohort still purchasing",
                 color=NAVY, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Months since acquisition", color=BODY_GRAY, fontsize=10)
    fig.colorbar(im, ax=ax, label="Retention (%)", shrink=0.85)
    fig.text(0.5, -0.02,
             "Blank cells are not yet observed: later cohorts have had less time. "
             "Month 0 is 100% by definition.",
             ha="center", fontsize=8.5, color="#777777")
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "cohort_retention.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------

def export(retention, active, sizes, per_member, cumulative):
    retention.round(2).to_csv(WIDE_FILE)

    rows = []
    for cohort in retention.index:
        for offset in retention.columns:
            val = retention.loc[cohort, offset]
            if pd.isna(val):
                continue
            rows.append({
                "cohort": str(cohort),
                "cohort_size": int(sizes[cohort]),
                "months_since_acquisition": int(offset),
                "active_customers": int(active.loc[cohort, offset]),
                "retention_pct": round(float(val), 2),
                "revenue_per_member": round(float(per_member.loc[cohort, offset]), 2),
                "cumulative_revenue_per_member": round(
                    float(cumulative.loc[cohort, offset]), 2),
                "is_observed": True,
            })
    pd.DataFrame(rows).to_csv(LONG_FILE, index=False)


def main():
    sales = load_sales()
    sales, excluded = drop_partial_tail(sales)
    sales = build_cohorts(sales)

    last_period = sales["month"].max()
    retention, active, sizes, mask = retention_matrix(sales, last_period)
    per_member, cumulative = revenue_matrix(sales, sizes, mask)

    valid_index = report_sizes(sizes, excluded)
    report_matrix(retention, sizes, "RETENTION BY COHORT (% of cohort purchasing)")
    report_average_curve(retention, valid_index)
    report_cohort_quality(retention, valid_index, offset=1)
    report_revenue(cumulative, sizes, valid_index)

    export(retention, active, sizes, per_member, cumulative)
    path = save_heatmap(retention, sizes, valid_index)

    print(f"\nWrote {WIDE_FILE}")
    print(f"Wrote {LONG_FILE}  (tidy format for Power BI)")
    print(f"Chart -> {path}")


if __name__ == "__main__":
    main()
