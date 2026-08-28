"""
scenario_simulator.py

A "what-if" tool for testing retention/discount campaigns against the RFM
segments before spending money on them.

The question it answers is not "what will happen" -- nobody knows that -- but
"given these assumptions, does this campaign pay for itself, and how wrong can
my assumptions be before it stops?" Assumptions are inputs, not estimates, so
this tool does not depend on the transaction data containing predictive
signal.

Two things it does that naive ROI models usually get wrong:

  1. Discount cannibalization. If you offer 15% off to a segment, you pay that
     discount to everyone who redeems -- including the customers who would have
     bought anyway. Ignoring this is the single most common way campaign ROI
     gets overstated.

  2. Break-even lift. Rather than only reporting ROI for one guessed lift, it
     solves for the lift the campaign would need to break even, which is the
     number that actually informs a go/no-go decision.

Usage:
    python scripts/scenario_simulator.py --list
    python scripts/scenario_simulator.py --segment "At Risk / Can't Lose Them" \
        --discount 15 --lift 10
    python scripts/scenario_simulator.py --segment Champions --sweep

Inputs:
    data/rfm_segments.csv, data/raw_sales_data.csv
"""

import argparse
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
CHART_DIR = os.path.join(REPORTS_DIR, "charts")

RFM_FILE = os.path.join(DATA_DIR, "rfm_segments.csv")
SALES_FILE = os.path.join(DATA_DIR, "raw_sales_data.csv")

HORIZON_DAYS = 30

# Home goods gross margin. A planning assumption, not measured -- the catalog
# has retail prices only, no cost of goods.
DEFAULT_GROSS_MARGIN = 0.45
DEFAULT_REDEMPTION = 0.80   # share of purchasers who actually use the offer
DEFAULT_CONTACT_COST = 0.35  # per targeted customer (email-scale)

NAVY = "#1F3864"
GOLD = "#C9973E"


# --------------------------------------------------------------------------
# baselines measured out of sample
# --------------------------------------------------------------------------

def measure_baselines(horizon=HORIZON_DAYS):
    """Forward-looking conversion rate and order value per segment.

    Segments are rebuilt on history only, then conversion is measured in a
    held-out window. Measuring conversion on the same window that defined
    `recency` would be circular -- "At Risk" means "has not purchased
    recently", so it would show a 0% baseline by construction and make every
    campaign look infinitely profitable.
    """
    sales = pd.read_csv(SALES_FILE, parse_dates=["InvoiceDate"])
    sales["Rev"] = sales["UnitPrice"] * sales["Quantity"]

    cutoff = sales["InvoiceDate"].max() - pd.Timedelta(days=horizon)
    hist = sales[sales["InvoiceDate"] <= cutoff]
    future_buyers = set(sales[sales["InvoiceDate"] > cutoff]["CustomerID"])

    snapshot = hist["InvoiceDate"].max() + pd.Timedelta(days=1)
    grouped = hist.groupby("CustomerID")
    rfm = pd.DataFrame({
        "recency": (snapshot - grouped["InvoiceDate"].max()).dt.days,
        "frequency": grouped["InvoiceNo"].nunique(),
        "monetary": grouped["Rev"].sum(),
    })
    rfm["r"] = pd.qcut(rfm["recency"].rank(method="first", ascending=False),
                       5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["f"] = pd.qcut(rfm["frequency"].rank(method="first"),
                       5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m"] = pd.qcut(rfm["monetary"].rank(method="first"),
                       5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["segment"] = rfm.apply(_segment_of, axis=1)
    rfm["converted"] = rfm.index.isin(future_buyers)
    rfm["aov"] = grouped.apply(
        lambda g: g["Rev"].sum() / g["InvoiceNo"].nunique(), include_groups=False)

    out = rfm.groupby("segment").agg(
        observed_customers=("converted", "size"),
        baseline_conv=("converted", "mean"),
        aov=("aov", "mean"),
    )
    return out, rfm["converted"].mean()


def _segment_of(row):
    r, f, m = row["r"], row["f"], row["m"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customers"
    if r >= 3 and f <= 2:
        return "Promising / New"
    if r <= 2 and f >= 3:
        return "At Risk / Can't Lose Them"
    if r <= 2 and f <= 2:
        return "Hibernating / Lost"
    return "About to Sleep / Average"


def load_segments():
    if not os.path.exists(RFM_FILE) or not os.path.exists(SALES_FILE):
        print("Error: need both data/rfm_segments.csv and data/raw_sales_data.csv.")
        print("Run rfm_analysis.py (and simulator.py before it) first.")
        sys.exit(1)
    rfm = pd.read_csv(RFM_FILE)
    sizes = rfm["customer_segment"].value_counts()
    baselines, overall = measure_baselines()
    return sizes, baselines, overall


# --------------------------------------------------------------------------
# the economics
# --------------------------------------------------------------------------

def run_scenario(*, targeted, baseline_conv, aov, discount, lift,
                 gross_margin=DEFAULT_GROSS_MARGIN,
                 redemption=DEFAULT_REDEMPTION,
                 contact_cost=DEFAULT_CONTACT_COST):
    """Project the profit impact of a discount campaign.

    `lift` is relative: 0.10 means a 10% increase in the conversion rate
    (38% -> 41.8%), not 10 percentage points. Relative lift is how uplift is
    normally quoted in campaign results.
    """
    baseline_buyers = targeted * baseline_conv
    treated_conv = min(baseline_conv * (1 + lift), 1.0)
    total_buyers = targeted * treated_conv
    incremental_buyers = total_buyers - baseline_buyers

    redeemers = total_buyers * redemption

    # Baseline margin: what you'd have earned doing nothing.
    margin_baseline = baseline_buyers * aov * gross_margin

    # Campaign margin: redeemers earn a reduced margin, everyone else full.
    margin_campaign = (
        redeemers * aov * (gross_margin - discount)
        + (total_buyers - redeemers) * aov * gross_margin
    )

    discount_spend = redeemers * aov * discount
    contact_spend = targeted * contact_cost
    total_spend = discount_spend + contact_spend

    incremental_profit = margin_campaign - margin_baseline - contact_spend
    roi = incremental_profit / total_spend if total_spend else float("nan")

    return {
        "targeted": targeted,
        "baseline_conv": baseline_conv,
        "treated_conv": treated_conv,
        "baseline_buyers": baseline_buyers,
        "incremental_buyers": incremental_buyers,
        "incremental_revenue": incremental_buyers * aov,
        "cannibalized_discount": baseline_buyers * redemption * aov * discount,
        "discount_spend": discount_spend,
        "contact_spend": contact_spend,
        "total_spend": total_spend,
        "incremental_profit": incremental_profit,
        "roi": roi,
        "breakeven_lift": breakeven_lift(
            targeted=targeted, baseline_conv=baseline_conv, aov=aov,
            discount=discount, gross_margin=gross_margin,
            redemption=redemption, contact_cost=contact_cost),
    }


def breakeven_lift(*, targeted, baseline_conv, aov, discount,
                   gross_margin=DEFAULT_GROSS_MARGIN,
                   redemption=DEFAULT_REDEMPTION,
                   contact_cost=DEFAULT_CONTACT_COST):
    """Relative lift at which incremental profit is exactly zero.

    Closed form. With B baseline buyers, I incremental buyers, margin g,
    discount d, redemption r, contact spend C:

        profit = I*aov*g - (B+I)*r*aov*d - C = 0
        =>  I = (B*r*aov*d + C) / (aov*(g - r*d))

    and relative lift is I/B. If the per-order margin after discount is
    non-positive the campaign can never break even, however large the lift.
    """
    baseline_buyers = targeted * baseline_conv
    denom = aov * (gross_margin - redemption * discount)
    if denom <= 0 or baseline_buyers <= 0:
        return float("inf")
    needed = (baseline_buyers * redemption * aov * discount
              + targeted * contact_cost) / denom
    return needed / baseline_buyers


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------

def money(v):
    return f"${v:,.0f}" if abs(v) >= 1000 else f"${v:,.2f}"


def print_scenario(segment, r, *, discount, lift, gross_margin, redemption,
                   contact_cost):
    line = "-" * 66
    print(f"\n{line}\nSCENARIO: {segment}\n{line}")
    print(f"  Targeted customers        {r['targeted']:>14,}")
    print(f"  Baseline conversion       {r['baseline_conv']*100:>13.1f}%   "
          f"(no campaign)")
    print(f"  Assumed lift              {lift*100:>13.1f}%   "
          f"-> {r['treated_conv']*100:.1f}% conversion")
    print(f"  Discount offered          {discount*100:>13.1f}%")
    print(f"  Gross margin              {gross_margin*100:>13.1f}%   (assumed)")
    print(f"  Redemption rate           {redemption*100:>13.1f}%   (assumed)")

    print(f"\n  Buyers without campaign   {r['baseline_buyers']:>14.1f}")
    print(f"  Incremental buyers        {r['incremental_buyers']:>14.1f}")
    print(f"  Incremental revenue       {money(r['incremental_revenue']):>14}")

    print(f"\n  COSTS")
    print(f"    Discount to incremental {money(r['discount_spend'] - r['cannibalized_discount']):>12}")
    print(f"    Discount cannibalized   {money(r['cannibalized_discount']):>12}"
          f"   <- paid to buyers you'd have had anyway")
    print(f"    Contact cost            {money(r['contact_spend']):>12}")
    print(f"    Total spend             {money(r['total_spend']):>12}")

    verdict = "PROFITABLE" if r["incremental_profit"] > 0 else "LOSES MONEY"
    print(f"\n  Incremental profit        {money(r['incremental_profit']):>14}")
    print(f"  ROI                       {r['roi']*100:>13.1f}%   {verdict}")

    be = r["breakeven_lift"]
    if be == float("inf"):
        print(f"  Break-even lift                     never   "
              f"discount exceeds margin")
    else:
        margin_note = "achievable" if be < 0.5 else "demanding"
        print(f"  Break-even lift           {be*100:>13.1f}%   "
              f"needed to justify spend ({margin_note})")
    print(line)


def sweep(segment, targeted, baseline_conv, aov, *, gross_margin, redemption,
          contact_cost, discounts=(5, 10, 15, 20, 25), lifts=(5, 10, 20, 30, 50)):
    """ROI grid across discount x lift, the sensitivity half of the tool."""
    rows = []
    for d in discounts:
        row = {}
        for l in lifts:
            r = run_scenario(targeted=targeted, baseline_conv=baseline_conv,
                             aov=aov, discount=d / 100, lift=l / 100,
                             gross_margin=gross_margin, redemption=redemption,
                             contact_cost=contact_cost)
            row[f"lift {l}%"] = r["roi"] * 100
        rows.append(pd.Series(row, name=f"disc {d}%"))
    grid = pd.DataFrame(rows)

    print(f"\nROI (%) SENSITIVITY -- {segment}")
    print("  rows = discount offered, cols = assumed conversion lift")
    print("  negative = campaign destroys value\n")
    print(grid.round(0).to_string())
    print("\n  break-even lift by discount:")
    for d in discounts:
        be = breakeven_lift(targeted=targeted, baseline_conv=baseline_conv,
                            aov=aov, discount=d / 100, gross_margin=gross_margin,
                            redemption=redemption, contact_cost=contact_cost)
        txt = "never (discount exceeds margin)" if be == float("inf") else f"{be*100:.1f}%"
        print(f"    {d:>2}% discount -> {txt}")
    return grid


def save_heatmap(grid, segment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(CHART_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = grid.values
    # Diverging scale pinned at 0 (the profit/loss boundary) but stretched to
    # each side's real range -- a symmetric scale lets one large positive ROI
    # wash out all the negative cells.
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    norm = TwoSlopeNorm(vmin=min(lo, -1e-9), vcenter=0.0, vmax=max(hi, 1e-9))
    im = ax.imshow(vals, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns, fontsize=9)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=9)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.0f}%", ha="center", va="center",
                    fontsize=9, color="#222222")
    ax.set_title(f"Campaign ROI sensitivity — {segment}", color=NAVY,
                 fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, label="ROI (%)")
    fig.tight_layout()
    safe = segment.replace(" / ", "_").replace(" ", "_").replace("'", "")
    path = os.path.join(CHART_DIR, f"scenario_roi_{safe}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="What-if campaign scenario simulator")
    p.add_argument("--segment", help="RFM segment to target")
    p.add_argument("--discount", type=float, default=15, help="discount %% offered")
    p.add_argument("--lift", type=float, default=10,
                   help="assumed relative %% lift in conversion")
    p.add_argument("--baseline", type=float,
                   help="override baseline conversion %% (recommended)")
    p.add_argument("--aov", type=float, help="override average order value")
    p.add_argument("--margin", type=float, default=DEFAULT_GROSS_MARGIN * 100,
                   help="gross margin %%")
    p.add_argument("--redemption", type=float, default=DEFAULT_REDEMPTION * 100,
                   help="%% of buyers who redeem the offer")
    p.add_argument("--contact-cost", type=float, default=DEFAULT_CONTACT_COST,
                   help="cost per targeted customer")
    p.add_argument("--sweep", action="store_true", help="show ROI sensitivity grid")
    p.add_argument("--chart", action="store_true", help="save sweep as a heatmap")
    p.add_argument("--list", action="store_true", help="list segments and exit")
    args = p.parse_args()

    sizes, baselines, overall = load_segments()

    if args.list or not args.segment:
        print(f"\nSegments (baseline = forward-looking {HORIZON_DAYS}-day "
              f"conversion, measured out of sample):\n")
        print(f"  {'segment':<28}{'customers':>10}{'baseline':>10}{'AOV':>12}")
        for seg in sizes.index:
            b = baselines.loc[seg] if seg in baselines.index else None
            if b is None:
                continue
            print(f"  {seg:<28}{sizes[seg]:>10,}{b.baseline_conv*100:>9.1f}%"
                  f"{money(b.aov):>12}")
        print(f"\n  overall baseline conversion: {overall*100:.1f}%")
        print("\n  NOTE: in this dataset the segment baselines above are"
              " statistically\n  indistinguishable from each other -- transactions are"
              " assigned to\n  customers at random by simulator.py, so past behavior does"
              " not\n  predict future purchases. Pass --baseline with your own estimate"
              "\n  for any real decision.")
        if not args.segment:
            return

    seg = args.segment
    if seg not in sizes.index:
        print(f"Error: unknown segment {seg!r}. Use --list to see valid names.")
        sys.exit(1)

    targeted = int(sizes[seg])
    base = (args.baseline / 100 if args.baseline is not None
            else float(baselines.loc[seg, "baseline_conv"]))
    aov = args.aov if args.aov is not None else float(baselines.loc[seg, "aov"])
    margin, redemption = args.margin / 100, args.redemption / 100

    r = run_scenario(targeted=targeted, baseline_conv=base, aov=aov,
                     discount=args.discount / 100, lift=args.lift / 100,
                     gross_margin=margin, redemption=redemption,
                     contact_cost=args.contact_cost)
    print_scenario(seg, r, discount=args.discount / 100, lift=args.lift / 100,
                   gross_margin=margin, redemption=redemption,
                   contact_cost=args.contact_cost)

    if args.sweep or args.chart:
        grid = sweep(seg, targeted, base, aov, gross_margin=margin,
                     redemption=redemption, contact_cost=args.contact_cost)
        if args.chart:
            print(f"\n  heatmap -> {save_heatmap(grid, seg)}")


if __name__ == "__main__":
    main()
