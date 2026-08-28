"""
generate_report.py

Builds reports/Wayfair_Sales_Performance_Report.docx from the pipeline's
CSV outputs.

This script is the missing link in the pipeline: the original report was
authored interactively and its generating code was never saved, which meant
the report could not be rebuilt when upstream data changed. Everything here
is derived from the CSVs at runtime -- there are no hardcoded figures in the
narrative text. If the simulation window, catalog, or segment logic changes,
re-running this script produces a report whose prose still matches its data.

Inputs:
    data/product_catalog_clean.csv  (real scraped catalog)
    data/raw_sales_data.csv         (simulated transactions)
    data/rfm_segments.csv           (optional; adds the segmentation section)

Output:
    reports/Wayfair_Sales_Performance_Report.docx
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
CHART_DIR = os.path.join(REPORTS_DIR, "charts")

SALES_FILE = os.path.join(DATA_DIR, "raw_sales_data.csv")
CATALOG_FILE = os.path.join(DATA_DIR, "product_catalog_clean.csv")
RFM_FILE = os.path.join(DATA_DIR, "rfm_segments.csv")
OUTPUT_FILE = os.path.join(REPORTS_DIR, "Wayfair_Sales_Performance_Report.docx")

# Palette lifted from the original report so regenerated output stays visually
# identical to the version that was signed off on.
NAVY = "#1F3864"
GOLD = "#C9973E"
BODY_GRAY = "#444444"
FILL_TINT = "#EDF0F4"
NAVY_RGB = RGBColor(0x1F, 0x38, 0x64)
GOLD_RGB = RGBColor(0xC9, 0x97, 0x3E)
BODY_RGB = RGBColor(0x44, 0x44, 0x44)

CHART_WIDTH_IN = 6.04
CHART_DPI = 180

# Segment display order: strongest to weakest, so tables read as a funnel
# rather than in the arbitrary order groupby happens to return.
SEGMENT_ORDER = [
    "Champions",
    "Loyal Customers",
    "Promising / New",
    "About to Sleep / Average",
    "At Risk / Can't Lose Them",
    "Hibernating / Lost",
]


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

def money(value: float) -> str:
    return f"${value:,.2f}"


def money_short(value: float) -> str:
    """Compact currency for prose, e.g. $92K."""
    return f"${value / 1000:,.0f}K"


def style_chart_axes(ax) -> None:
    """Apply the report's chart look: no box, faint horizontal guides only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=BODY_GRAY, labelsize=10)


def save_chart(fig, filename: str) -> str:
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def shade_cell(cell, hex_fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill.lstrip("#"))
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_text(cell, text: str, *, bold=False, color=BODY_RGB, size=10, align=None):
    cell.text = ""
    para = cell.paragraphs[0]
    if align is not None:
        para.alignment = align
    run = para.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = color


def add_table(doc, headers, rows, widths=None, numeric_from=1):
    """Table with a navy header band and zebra striping."""
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, head in enumerate(headers):
        cell = table.rows[0].cells[i]
        shade_cell(cell, NAVY)
        set_cell_text(
            cell, head, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), size=9,
            align=WD_ALIGN_PARAGRAPH.RIGHT if i >= numeric_from else None,
        )

    for r_idx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            if r_idx % 2 == 1:
                shade_cell(cells[i], FILL_TINT)
            set_cell_text(
                cells[i], str(value), size=9,
                align=WD_ALIGN_PARAGRAPH.RIGHT if i >= numeric_from else None,
            )

    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    return table


def add_body(doc, text: str, *, bold=False, size=10.5, space_after=8):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(space_after)
    run = para.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = BODY_RGB
    return para


def add_bullets(doc, items):
    for item in items:
        para = doc.add_paragraph(style="List Paragraph")
        para.paragraph_format.space_after = Pt(4)
        run = para.add_run(item)
        run.font.size = Pt(10.5)
        run.font.color.rgb = BODY_RGB


def add_heading(doc, text: str, level: int):
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = NAVY_RGB
    return heading


def add_chart(doc, path: str):
    doc.add_picture(path, width=Inches(CHART_WIDTH_IN))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# --------------------------------------------------------------------------
# data loading + metrics
# --------------------------------------------------------------------------

def load_data():
    for path, label in ((SALES_FILE, "sales data"), (CATALOG_FILE, "product catalog")):
        if not os.path.exists(path):
            print(f"Error: {label} not found at {path}.")
            print("Run scraper.py -> clean_data.py -> simulator.py first.")
            sys.exit(1)

    sales = pd.read_csv(SALES_FILE)
    sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])
    sales["Revenue"] = sales["UnitPrice"] * sales["Quantity"]
    catalog = pd.read_csv(CATALOG_FILE)

    # rfm_segments.csv is optional: the report degrades to its pre-segmentation
    # form rather than failing if rfm_analysis.py hasn't been run yet.
    rfm = pd.read_csv(RFM_FILE) if os.path.exists(RFM_FILE) else None
    return sales, catalog, rfm


def compute_kpis(sales, catalog):
    orders_per_customer = sales.groupby("CustomerID")["InvoiceNo"].nunique()
    total_orders = sales["InvoiceNo"].nunique()
    total_revenue = sales["Revenue"].sum()
    repeat_rate = (orders_per_customer >= 2).mean() * 100

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_units": int(sales["Quantity"].sum()),
        "avg_order_value": total_revenue / total_orders,
        "unique_customers": sales["CustomerID"].nunique(),
        "products_sold": sales["ProductID"].nunique(),
        "catalog_size": len(catalog),
        "repeat_rate": repeat_rate,
        "single_order_customers": int((orders_per_customer == 1).sum()),
        "orders_per_customer": orders_per_customer,
        "date_min": sales["InvoiceDate"].min(),
        "date_max": sales["InvoiceDate"].max(),
    }


def product_performance(sales):
    perf = sales.groupby(["ProductID", "ProductName"]).agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"),
    ).reset_index().sort_values("Revenue", ascending=False)
    perf["RevenueShare"] = perf["Revenue"] / perf["Revenue"].sum() * 100
    perf["CumulativeShare"] = perf["RevenueShare"].cumsum()
    return perf


def monthly_revenue(sales):
    """Monthly totals with any partial trailing month dropped.

    The simulation window ends on a month boundary, so the final month holds
    a single day of data and would otherwise render as a false cliff in the
    trend chart. Detected from the data rather than hardcoded.
    """
    monthly = sales.groupby(sales["InvoiceDate"].dt.to_period("M")).agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
    )
    last_period = monthly.index.max()
    days_in_last = sales.loc[
        sales["InvoiceDate"].dt.to_period("M") == last_period, "InvoiceDate"
    ].dt.day.nunique()
    excluded = None
    if len(monthly) > 1 and days_in_last <= 5:
        excluded = last_period
        monthly = monthly.iloc[:-1]
    return monthly, excluded


def segment_summary(rfm):
    summary = rfm.groupby("customer_segment").agg(
        Customers=("CustomerID", "count"),
        Revenue=("monetary", "sum"),
        AvgFrequency=("frequency", "mean"),
        AvgRecency=("recency", "mean"),
    ).reset_index()
    summary["CustomerShare"] = summary["Customers"] / summary["Customers"].sum() * 100
    summary["RevenueShare"] = summary["Revenue"] / summary["Revenue"].sum() * 100
    summary["AvgValue"] = summary["Revenue"] / summary["Customers"]

    present = [s for s in SEGMENT_ORDER if s in set(summary["customer_segment"])]
    extras = [s for s in summary["customer_segment"] if s not in SEGMENT_ORDER]
    order = present + sorted(extras)
    summary["_sort"] = summary["customer_segment"].map({s: i for i, s in enumerate(order)})
    return summary.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------

def chart_monthly_revenue(monthly):
    labels = [p.strftime("%b %Y") for p in monthly.index]
    fig, ax = plt.subplots(figsize=(10, 4.65))
    ax.plot(labels, monthly["Revenue"], color=NAVY, linewidth=2.6, zorder=3)
    ax.scatter(labels, monthly["Revenue"], color=GOLD, s=62, zorder=4,
               edgecolors="white", linewidths=1.1)
    ax.fill_between(labels, monthly["Revenue"], color=NAVY, alpha=0.06, zorder=2)
    ax.set_title(f"Monthly Revenue Trend ({labels[0]} \u2013 {labels[-1]})",
                 color=NAVY, fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Revenue ($)", color=BODY_GRAY, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    style_chart_axes(ax)
    return save_chart(fig, "monthly_revenue.png")


def chart_top_products(perf, top_n=10):
    top = perf.head(top_n).iloc[::-1]
    labels = [n if len(n) <= 42 else n[:39] + "..." for n in top["ProductName"]]
    # Highlight the single largest earner: the concentration story is the
    # report's headline finding, so the chart should show it at a glance.
    colors = [NAVY] * len(top)
    colors[-1] = GOLD
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels, top["Revenue"], color=colors, height=0.72)
    ax.set_title(f"Top {top_n} Products by Revenue", color=NAVY,
                 fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Revenue ($)", color=BODY_GRAY, fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.yaxis.grid(False)
    style_chart_axes(ax)
    ax.tick_params(axis="y", labelsize=9)
    return save_chart(fig, "top_products.png")


def chart_pareto(perf):
    ranks = range(1, len(perf) + 1)
    fig, ax = plt.subplots(figsize=(10, 4.65))
    ax.bar(ranks, perf["Revenue"], color=NAVY, width=0.75)
    ax.set_ylabel("Revenue ($)", color=NAVY, fontsize=11)
    ax.set_xlabel("Products, ranked by revenue (1 = highest)",
                  color=BODY_GRAY, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    style_chart_axes(ax)
    ax.tick_params(axis="y", colors=NAVY)

    ax2 = ax.twinx()
    ax2.plot(ranks, perf["CumulativeShare"], color=GOLD, linewidth=2.4,
             marker="o", markersize=4)
    ax2.axhline(80, color="#AAAAAA", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("Cumulative Share of Revenue (%)", color=GOLD, fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors=GOLD, labelsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    ax.set_title("Revenue Concentration Across Product Catalog", color=NAVY,
                 fontsize=17, fontweight="bold", pad=16)
    return save_chart(fig, "revenue_concentration.png")


def chart_customer_distribution(orders_per_customer):
    counts = orders_per_customer.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(counts.index, counts.values, color=NAVY, width=0.72)
    ax.set_title("Distribution of Orders per Customer", color=NAVY,
                 fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Orders placed", color=BODY_GRAY, fontsize=11)
    ax.set_ylabel("Number of customers", color=BODY_GRAY, fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    style_chart_axes(ax)
    return save_chart(fig, "customer_distribution.png")


def chart_segments(summary):
    """Customer count vs revenue share, side by side.

    Paired deliberately: the gap between a segment's share of customers and
    its share of revenue is the actionable part of an RFM breakdown.
    """
    labels = [s.replace(" / ", " /\n") for s in summary["customer_segment"]][::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)

    ax1.barh(labels, summary["Customers"][::-1], color=NAVY, height=0.7)
    ax1.set_title("Customers per Segment", color=NAVY, fontsize=13,
                  fontweight="bold", pad=12)
    ax1.set_xlabel("Customers", color=BODY_GRAY, fontsize=10)
    ax1.xaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    ax1.yaxis.grid(False)
    style_chart_axes(ax1)
    ax1.tick_params(axis="y", labelsize=8.5)

    ax2.barh(labels, summary["Revenue"][::-1], color=GOLD, height=0.7)
    ax2.set_title("Revenue per Segment", color=GOLD, fontsize=13,
                  fontweight="bold", pad=12)
    ax2.set_xlabel("Revenue ($)", color=BODY_GRAY, fontsize=10)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:,.0f}K"))
    ax2.xaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    ax2.yaxis.grid(False)
    style_chart_axes(ax2)

    fig.suptitle("Customer Segmentation (RFM)", color=NAVY, fontsize=17,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_chart(fig, "rfm_segments.png")


# --------------------------------------------------------------------------
# document assembly
# --------------------------------------------------------------------------

def build_document(sales, catalog, rfm):
    kpis = compute_kpis(sales, catalog)
    perf = product_performance(sales)
    monthly, excluded_period = monthly_revenue(sales)
    summary = segment_summary(rfm) if rfm is not None else None

    doc = Document()
    section = doc.sections[0]
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BODY_RGB

    span = f"{kpis['date_min']:%b %Y} – {kpis['date_max']:%b %Y}"

    # --- title block -------------------------------------------------------
    title = doc.add_paragraph()
    t_run = title.add_run("Wayfair Product & Sales Performance Report")
    t_run.bold = True
    t_run.font.size = Pt(22)
    t_run.font.color.rgb = NAVY_RGB

    sub = doc.add_paragraph()
    s_run = sub.add_run(f"E-Commerce Analytics Project — Annual Review ({span})")
    s_run.font.size = Pt(13)
    s_run.font.color.rgb = GOLD_RGB

    add_heading(doc, "About This Report", 2)
    add_body(doc,
        "Product catalog data (names and prices) was scraped directly from live "
        "Wayfair product listing pages. Customer identities and the full transaction "
        "history (invoices, dates, and quantities) are synthetically generated, since "
        "Wayfair's actual sales and customer data is proprietary and not accessible "
        "for this analysis. All customer names appearing in this report are "
        "computer-generated and do not refer to real individuals. Findings on "
        "product-level performance reflect real catalog data; findings on revenue "
        "totals, trends, and customer behavior reflect the simulated transaction "
        "layer built on top of it.")

    # --- executive summary -------------------------------------------------
    add_heading(doc, "Executive Summary", 1)
    add_body(doc,
        f"Over the trailing twelve-month period, the simulated transaction history "
        f"recorded {kpis['total_orders']:,} orders across {kpis['unique_customers']:,} "
        f"customers and {kpis['products_sold']} catalog products, totaling "
        f"{money(kpis['total_revenue'])} in revenue. The analysis below breaks this "
        f"down by time, product, and customer to surface where revenue concentrates "
        f"and how purchasing behavior is distributed.")

    add_table(doc,
        ["Metric", "Value"],
        [
            ("Total Revenue", money(kpis["total_revenue"])),
            ("Total Orders", f"{kpis['total_orders']:,}"),
            ("Total Units Sold", f"{kpis['total_units']:,}"),
            ("Average Order Value", money(kpis["avg_order_value"])),
            ("Unique Customers", f"{kpis['unique_customers']:,}"),
            ("Products Sold",
             f"{kpis['products_sold']} of {kpis['catalog_size']} catalog items"),
            ("Repeat Customer Rate", f"{kpis['repeat_rate']:.1f}%"),
        ],
        widths=[3.4, 3.6])
    doc.add_paragraph()

    top_product = perf.iloc[0]
    top_price = sales.loc[sales["ProductID"] == top_product["ProductID"], "UnitPrice"].max()
    top5_share = perf.head(5)["RevenueShare"].sum()
    peak_month = monthly["Revenue"].idxmax()
    trough_month = monthly["Revenue"].idxmin()

    add_body(doc, "Key observations:", bold=True)
    add_bullets(doc, [
        f"A single product — a {money(top_price)} item — accounts for "
        f"{top_product['RevenueShare']:.1f}% of total revenue on its own, despite "
        f"representing just 1 of {kpis['catalog_size']} catalog items.",

        f"The top 5 products by revenue account for {top5_share:.1f}% of all revenue; "
        f"the remaining {kpis['catalog_size'] - 5} products share the rest.",

        f"{kpis['repeat_rate']:.1f}% of customers placed more than one order, indicating "
        f"strong simulated repeat-purchase behavior across the customer base.",

        f"Monthly revenue is relatively stable, ranging from roughly "
        f"{money_short(monthly['Revenue'].min())} to "
        f"{money_short(monthly['Revenue'].max())} per month with no sustained downward "
        f"trend across the {len(monthly)} full months of data.",
    ])

    # --- revenue trend -----------------------------------------------------
    add_heading(doc, "Sales & Revenue Trends", 1)
    excluded_note = ""
    if excluded_period is not None:
        excluded_note = (
            f" {excluded_period.strftime('%B %Y')} is excluded from this chart because "
            f"the underlying transaction window ends on {kpis['date_max']:%Y-%m-%d}, so "
            f"that month contains only a single day of data and would otherwise appear "
            f"as a misleading drop-off.")
    add_body(doc,
        f"The chart below shows monthly revenue across the {len(monthly)} full months "
        f"in the dataset ({monthly.index.min().strftime('%B %Y')} through "
        f"{monthly.index.max().strftime('%B %Y')}).{excluded_note}")
    add_chart(doc, chart_monthly_revenue(monthly))
    add_body(doc,
        f"Revenue peaked in {peak_month.strftime('%B %Y')} at "
        f"${monthly['Revenue'].max():,.0f}, consistent with a holiday shopping pattern, "
        f"and dipped to its lowest full-month point in "
        f"{trough_month.strftime('%B %Y')} at ${monthly['Revenue'].min():,.0f}. Order "
        f"counts follow a similar rhythm, ranging from {monthly['Orders'].min()} to "
        f"{monthly['Orders'].max()} orders per month. Overall, the trend shows healthy "
        f"month-to-month variation without a sustained decline.")

    # --- product performance ----------------------------------------------
    add_heading(doc, "Product Performance", 1)
    add_body(doc, "The chart below ranks the ten highest-revenue products. "
                  "The top performer is highlighted.")
    add_chart(doc, chart_top_products(perf))

    def product_rows(frame):
        return [
            (r["ProductName"][:52], money(r["Revenue"]),
             f"{int(r['Units']):,}", f"{int(r['Orders']):,}")
            for _, r in frame.iterrows()
        ]

    add_heading(doc, "Top 10 Products by Revenue", 2)
    add_table(doc, ["Product", "Revenue", "Units Sold", "Orders"],
              product_rows(perf.head(10)), widths=[3.5, 1.3, 1.1, 0.9])
    doc.add_paragraph()

    bottom5 = perf.tail(5).iloc[::-1]
    add_heading(doc, "Bottom 5 Products by Revenue", 2)
    add_table(doc, ["Product", "Revenue", "Units Sold", "Orders"],
              product_rows(bottom5), widths=[3.5, 1.3, 1.1, 0.9])
    doc.add_paragraph()

    add_body(doc,
        f"Notably, order counts for the lowest-revenue products "
        f"({int(bottom5['Orders'].min())}–{int(bottom5['Orders'].max())} orders each) "
        f"are comparable to those of the highest-revenue products — the gap is driven "
        f"almost entirely by unit price, not by how often the product sells. "
        f"Low-priced, frequently-selling items are moving consistent volume but "
        f"contributing comparatively little revenue per transaction.")

    # --- concentration -----------------------------------------------------
    products_to_80 = int((perf["CumulativeShare"] < 80).sum() + 1)
    add_heading(doc, "Revenue Concentration", 1)
    add_body(doc,
        f"The chart below ranks all {len(perf)} products from highest to lowest revenue "
        f"(bars, left axis) alongside the cumulative share of total revenue they "
        f"represent (line, right axis). The dashed line marks the 80% cumulative "
        f"revenue threshold.")
    add_chart(doc, chart_pareto(perf))
    add_body(doc,
        f"Revenue is meaningfully concentrated at the top of the catalog: it takes only "
        f"{products_to_80} of {len(perf)} products "
        f"({products_to_80 / len(perf) * 100:.0f}% of the lineup) to reach 80% of total "
        f"revenue, driven disproportionately by a small number of higher-priced rugs. "
        f"This is a useful signal for inventory and marketing prioritization — the "
        f"catalog's commercial performance is not evenly distributed across products.")

    # --- customer behavior -------------------------------------------------
    opc = kpis["orders_per_customer"]
    add_heading(doc, "Customer Behavior", 1)
    add_body(doc,
        f"The distribution below shows how many orders each of the "
        f"{kpis['unique_customers']:,} customers placed over the twelve-month period. "
        f"Most customers placed between {int(opc.quantile(0.25))} and "
        f"{int(opc.quantile(0.75))} orders; very few placed only one.")
    add_chart(doc, chart_customer_distribution(opc))
    add_body(doc,
        f"Only {kpis['single_order_customers']} of {kpis['unique_customers']:,} customers "
        f"({100 - kpis['repeat_rate']:.1f}%) placed a single order, while "
        f"{kpis['repeat_rate']:.1f}% placed two or more — a high repeat rate, "
        f"consistent with the simulation's design assumptions rather than a claim about "
        f"real Wayfair shopper behavior.")

    cust_rev = sales.groupby(["CustomerID", "CustomerName"]).agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
    ).reset_index().sort_values("Revenue", ascending=False).head(10)

    add_heading(doc, "Top 10 Customers by Revenue", 2)
    add_body(doc, "Customer names below are computer-generated and do not represent "
                  "real individuals.")
    add_table(doc, ["Customer", "Total Revenue", "Orders"],
        [(r["CustomerName"], money(r["Revenue"]), f"{int(r['Orders']):,}")
         for _, r in cust_rev.iterrows()],
        widths=[3.5, 1.9, 1.4])
    doc.add_paragraph()

    context = {
        "kpis": kpis,
        "perf": perf,
        "monthly": monthly,
        "summary": summary,
        "top_product": top_product,
        "top5_share": top5_share,
        "products_to_80": products_to_80,
    }
    return doc, context


def add_segmentation_section(doc, summary, kpis):
    """RFM segmentation section.

    The previous version of this report closed by naming RFM segmentation as a
    natural next step; rfm_analysis.py now produces it, so it belongs in the
    report proper.
    """
    add_heading(doc, "Customer Segmentation (RFM)", 1)
    add_body(doc,
        "Each customer is scored on three dimensions — Recency (how recently they "
        "last ordered), Frequency (how many orders they placed), and Monetary value "
        "(how much they spent). Scores are assigned on a 1–5 quintile scale relative "
        "to the rest of the customer base, then combined into the segment labels "
        "below. Because scores are relative, every segment is defined against this "
        "population rather than an absolute benchmark.")
    add_chart(doc, chart_segments(summary))

    add_heading(doc, "Segment Breakdown", 2)
    add_table(doc,
        ["Segment", "Customers", "% of Base", "Revenue", "% of Revenue", "Avg Value"],
        [(r["customer_segment"], f"{int(r['Customers']):,}", f"{r['CustomerShare']:.1f}%",
          money(r["Revenue"]), f"{r['RevenueShare']:.1f}%", money(r["AvgValue"]))
         for _, r in summary.iterrows()],
        widths=[1.9, 1.0, 0.95, 1.35, 1.05, 1.15])
    doc.add_paragraph()

    top_rev = summary.loc[summary["Revenue"].idxmax()]
    champions = summary[summary["customer_segment"] == "Champions"]
    at_risk = summary[summary["customer_segment"] == "At Risk / Can't Lose Them"]
    lost = summary[summary["customer_segment"] == "Hibernating / Lost"]

    base_avg = summary["Revenue"].sum() / summary["Customers"].sum()
    lines = []
    if not champions.empty:
        c = champions.iloc[0]
        lead = "also the largest revenue contributor" if             top_rev["customer_segment"] == "Champions" else "the strongest segment"
        lines.append(
            f"Champions — high scores across all three dimensions, and {lead} — "
            f"number {int(c['Customers'])} customers ({c['CustomerShare']:.1f}% of the "
            f"base) and generate {c['RevenueShare']:.1f}% of revenue, at "
            f"{money(c['AvgValue'])} per customer against a base-wide average of "
            f"{money(base_avg)}.")
    if champions.empty or top_rev["customer_segment"] != "Champions":
        lines.append(
            f"The {top_rev['customer_segment']} segment contributes the most revenue "
            f"overall, accounting for {top_rev['RevenueShare']:.1f}% of revenue from "
            f"{top_rev['CustomerShare']:.1f}% of the customer base.")
    if not at_risk.empty:
        a = at_risk.iloc[0]
        lines.append(
            f"At Risk / Can't Lose Them holds {int(a['Customers'])} customers worth "
            f"{money(a['Revenue'])} ({a['RevenueShare']:.1f}% of revenue) who bought "
            f"frequently but have not returned recently — the segment where retention "
            f"effort has the most revenue at stake.")
    if not lost.empty:
        h = lost.iloc[0]
        lines.append(
            f"Hibernating / Lost accounts for {h['CustomerShare']:.1f}% of customers but "
            f"only {h['RevenueShare']:.1f}% of revenue, so reactivation spend here should "
            f"be weighed against its comparatively low return.")
    add_body(doc, " ".join(lines))

    add_body(doc,
        f"One caveat carries over from the simulation's design: because transactions are "
        f"assigned to customers at random, the {kpis['repeat_rate']:.1f}% repeat rate is "
        f"far higher than real big-ticket home goods retail would produce. Segment sizes "
        f"here therefore describe the simulated population, not expected real-world "
        f"customer distribution.", size=9.5)


def add_takeaways(doc, ctx):
    kpis, summary = ctx["kpis"], ctx["summary"]
    top_product, perf = ctx["top_product"], ctx["perf"]

    add_heading(doc, "Key Takeaways", 1)
    items = [
        f"Revenue is heavily weighted toward a small number of high-ticket products — "
        f"one rug alone accounts for {top_product['RevenueShare']:.1f}% of total revenue. "
        f"Any stockout or pricing change on this item would have an outsized impact on "
        f"overall performance.",

        f"The top 5 products ({5 / len(perf) * 100:.0f}% of the catalog) generate "
        f"{ctx['top5_share']:.1f}% of all revenue, while the bottom half of the catalog "
        f"contributes comparatively little despite similar order volumes — suggesting "
        f"unit price, not popularity, is the primary revenue driver.",

        f"Order volume is fairly stable month over month, with a clear seasonal peak in "
        f"{ctx['monthly']['Revenue'].idxmax().strftime('%B')}, typical of holiday-driven "
        f"home goods demand.",

        f"Simulated repeat-purchase behavior is very high ({kpis['repeat_rate']:.1f}%); in "
        f"a real dataset this would be worth validating, since real-world repeat rates "
        f"for big-ticket home goods like area rugs are often lower.",
    ]
    if summary is not None:
        top_rev = summary.loc[summary["Revenue"].idxmax()]
        items.append(
            f"RFM segmentation concentrates revenue as sharply as the product catalog "
            f"does: {top_rev['CustomerShare']:.1f}% of customers in the "
            f"{top_rev['customer_segment']} segment drive {top_rev['RevenueShare']:.1f}% "
            f"of revenue, pointing retention effort at a well-defined group.")
        items.append(
            "With segmentation in place, modeling seasonality more formally or tuning "
            "the simulator toward realistic repeat rates would be the natural next steps.")
    else:
        items.append(
            "This report is a foundation for further analysis — segmenting customers by "
            "recency/frequency/monetary value (RFM), or modeling seasonality more "
            "formally, would be natural next steps.")
    add_bullets(doc, items)


def add_footer_note(doc, has_rfm: bool):
    doc.add_paragraph()
    sources = ("product_catalog_clean.csv (scraped Wayfair catalog data), "
               "raw_sales_data.csv (simulated transaction history)")
    if has_rfm:
        sources += ", and rfm_segments.csv (derived customer segments)"
    para = doc.add_paragraph()
    run = para.add_run(
        f"Data sources: {sources}. Generated by scripts/generate_report.py as part of "
        f"the e-commerce analytics project pipeline; all figures in this report are "
        f"computed from those files at generation time.")
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    run.italic = True


def main() -> None:
    print("Generating sales performance report...")
    sales, catalog, rfm = load_data()
    print(f"Loaded {len(sales):,} transactions, {len(catalog)} catalog products.")
    if rfm is None:
        print(f"[Notice] {RFM_FILE} not found — the segmentation section will be "
              f"omitted. Run rfm_analysis.py to include it.")
    else:
        print(f"Loaded {len(rfm):,} scored customers for segmentation.")

    doc, ctx = build_document(sales, catalog, rfm)
    if ctx["summary"] is not None:
        add_segmentation_section(doc, ctx["summary"], ctx["kpis"])
    add_takeaways(doc, ctx)
    add_footer_note(doc, ctx["summary"] is not None)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    try:
        doc.save(OUTPUT_FILE)
    except PermissionError:
        print(f"\nError: could not write {OUTPUT_FILE}.")
        print("The file is most likely open in Word — close it and re-run.")
        sys.exit(1)

    print(f"Wrote {OUTPUT_FILE}")
    print(f"Charts written to {CHART_DIR}")


if __name__ == "__main__":
    main()
