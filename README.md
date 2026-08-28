# E-Commerce RFM Analytics Pipeline

An analytics pipeline that scrapes **real** Wayfair product catalog data,
layers **simulated** customer transactions on top of it (since Wayfair's
actual sales history is proprietary), and produces an RFM (Recency,
Frequency, Monetary) customer segmentation plus a formal sales performance
report.

This README exists so a fresh session (human or AI) can pick up the project
without re-deriving decisions that were already made and tested.

## Quick start

```bash
pip install -r requirements.txt

python scripts/scraper.py        # data/wayfair_page.html -> data/product_catalog.csv
python scripts/clean_data.py     # data/product_catalog.csv -> data/product_catalog_clean.csv
python scripts/simulator.py      # data/product_catalog_clean.csv -> data/raw_sales_data.csv
python scripts/rfm_analysis.py   # data/raw_sales_data.csv -> data/rfm_segments.csv
python scripts/generate_report.py # all of the above -> reports/*.docx
```

All five scripts resolve paths relative to their own file location
(`SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))`), so they run
correctly whether invoked from the project root or from inside `scripts/`.
Verified both ways.

## Folder structure

```
ecommerce-rfm-pipeline/
├── data/
│   ├── wayfair_page.html         # locally saved Wayfair page (scraper input)
│   ├── product_catalog.csv       # scraper.py output (raw scraped catalog)
│   ├── product_catalog_clean.csv # clean_data.py output (cleaned catalog)
│   ├── raw_sales_data.csv        # simulator.py output (simulated transactions)
│   └── rfm_segments.csv          # rfm_analysis.py output (customer segments)
├── scripts/
│   ├── scraper.py
│   ├── clean_data.py
│   ├── simulator.py
│   ├── rfm_analysis.py
│   └── generate_report.py
├── reports/
│   ├── Wayfair_Sales_Performance_Report.docx
│   └── charts/                   # PNGs embedded in the report (regenerated each run)
├── requirements.txt
└── README.md
```

## Data provenance — the most important design decision in this project

**Real data:** product names and prices in `product_catalog.csv` /
`product_catalog_clean.csv`, scraped from `wayfair_page.html`.

**Simulated data:** everything in `raw_sales_data.csv` and
`rfm_segments.csv` — customer identities, invoices, dates, quantities. This
is intentional: real Wayfair sales/customer data is proprietary, so
transactions are generated against the real catalog instead. Customer
names are Faker-generated and do not refer to real people.

To prevent these two categories from ever silently blending, every catalog
row carries a `Source` column (`"scraped"` or `"synthetic_fallback"`).
`simulator.py` checks this column before generating transactions and will
warn/prompt if it finds non-`"scraped"` rows. This exists because early
drafts of `scraper.py`/`clean_data.py` had fallback logic that silently
overwrote real scraped data with fabricated products under the same
filename — that failure mode is why the current versions never overwrite
real output with synthetic data, and always write synthetic/fallback data
to a clearly distinct filename instead (`product_catalog_SAMPLE.csv`).

## Script-by-script notes

### `scraper.py`
- **Primary strategy**: selects product titles/prices by exact CSS class
  match, verified against the real page. Titles are `<h2>` elements with
  class set `{_6o3atzbl, _6o3atz18y, _1lxwj2q1}`; prices are `<span>`
  elements with class set `{_6o3atzbl, _6o3atzc7, _6o3atz1d9, _6o3atz1bd}`.
  This gave 100% clean extraction (49/49 real products) vs. a much noisier
  text-content heuristic used earlier.
- **⚠️ Known fragility**: those class names look like hashed/atomic CSS
  from a CSS-in-JS build. They will likely change whenever Wayfair ships a
  new frontend build. If `scraper.py` suddenly reports 0 products, this is
  the first thing to check — re-inspect `wayfair_page.html` for the current
  class names (search for a known product title string, e.g. via
  `BeautifulSoup`, and check its parent element's `class` attribute).
- **Fallback strategy**: if class-based extraction finds nothing, falls
  back to a text-heuristic (badge-word filtering + longest-candidate
  selection) so the script degrades gracefully rather than failing outright.
- Product IDs are a stable hash of `(name, price)`, so re-running the
  scraper doesn't reassign IDs to the same products.

### `clean_data.py`
- Filters out rows where the name is actually a price string, or matches
  known UI/navigation junk substrings.
- Never fabricates replacement data. If cleaning leaves fewer than 15 rows,
  it exits with an error explaining likely causes rather than silently
  injecting invented products.
- Never overwrites `product_catalog.csv` — always writes to
  `product_catalog_clean.csv`.

### `simulator.py`
- Generates 500 synthetic customers and 2,500 synthetic transactions
  against the real catalog, over a 365-day window ending 2026-06-01.
- Includes ±5% random price variance per transaction (real sales data
  isn't a flat price applied retroactively).
- `RANDOM_SEED = 42` is set for reproducibility — remove it for fresh
  randomness on each run.
- Checks the catalog's `Source` column before proceeding (see provenance
  note above).

### `rfm_analysis.py`
- Computes Recency / Frequency / Monetary per customer from
  `raw_sales_data.csv`, scores each on a 1–5 quintile scale, and assigns a
  segment label (`Champions`, `Loyal Customers`, `Promising / New`,
  `At Risk / Can't Lose Them`, `Hibernating / Lost`, `About to Sleep /
  Average`).
- All three quintile scores use `.rank(method='first')` before `qcut` so
  that tied values (common with integer `frequency`) can never produce a
  "duplicate bin edge" crash. This was a real bug found and fixed — adding
  `duplicates='drop'` to `qcut` directly does **not** work when explicit
  `labels` are also passed (it raises a different error), so don't
  reintroduce that pattern.
- `snapshot_date` is derived from the data itself
  (`max(InvoiceDate) + 1 day`), not hardcoded, so it stays correct even if
  `simulator.py`'s date window changes later.

### `generate_report.py`
- Builds `reports/Wayfair_Sales_Performance_Report.docx` from
  `raw_sales_data.csv`, `product_catalog_clean.csv`, and (optionally)
  `rfm_segments.csv`. Charts are written to `reports/charts/` and embedded.
- **Every number in the narrative prose is computed at generation time.**
  There are no hardcoded figures in the sentences — month names, peak/trough
  revenue, the 80%-of-revenue product count, segment shares, and the
  partial-month exclusion note all derive from the data. Changing the
  simulation window or catalog produces a report whose prose still matches
  its own tables.
- `rfm_segments.csv` is optional. If it's missing the script prints a notice
  and emits the report without the segmentation section, rather than failing.
- Verified to reproduce every figure from the original hand-authored report
  (revenue, orders, AOV, repeat rate, top/bottom products, top customers).
- Exits with a clear message if the .docx is open in Word, which holds a
  write lock on the file.
- **Why this script exists**: the original report was authored interactively
  and its generating code was never saved, so the report couldn't be rebuilt
  when upstream data changed. This closes that gap.

## Known data characteristics (from the last full run)

- 49 real products, 495 simulated customers, 2,500 simulated transactions,
  $813,442 total simulated revenue.
- Revenue is heavily concentrated: one $3,399.99 rug alone accounts for
  ~29% of total revenue; the top 5 products account for ~50%.
- 96.2% of simulated customers are "repeat" customers (placed 2+ orders) —
  this is an artifact of the simulation's random-assignment design, not a
  claim about real Wayfair shopper behavior. Worth keeping in mind when
  interpreting `rfm_segments.csv`: segment sizes reflect the simulation's
  assumptions, not real-world purchase patterns.

## Report

`reports/Wayfair_Sales_Performance_Report.docx` — formal Word report,
generated by `scripts/generate_report.py`. Covers KPIs, monthly revenue
trend, top/bottom products, revenue concentration (Pareto), customer
purchase frequency, and **RFM customer segmentation**. States data
provenance (real catalog vs. simulated transactions) up front, and repeats
the simulation caveat inside the segmentation section so segment sizes
aren't read as real-world customer distribution.

Segment breakdown from the last run (sums to the $813,442.22 revenue total):

| Segment | Customers | % of base | Revenue | % of revenue |
|---|---|---|---|---|
| Champions | 75 | 15.2% | $201,101.04 | 24.7% |
| Loyal Customers | 88 | 17.8% | $193,584.29 | 23.8% |
| Promising / New | 92 | 18.6% | $80,670.32 | 9.9% |
| About to Sleep / Average | 42 | 8.5% | $28,121.51 | 3.5% |
| At Risk / Can't Lose Them | 92 | 18.6% | $183,820.34 | 22.6% |
| Hibernating / Lost | 106 | 21.4% | $126,144.72 | 15.5% |

## Natural next steps

- If the scraper ever reports 0 products, re-derive the CSS classes (see
  fragility note above) before assuming the page has no products.
- Consider whether `simulator.py`'s repeat-purchase rate should be tuned
  down to be more realistic for big-ticket home goods.
- Consider tuning `simulator.py`'s repeat-purchase rate (see above); it is
  the main thing standing between this dataset and a realistic one.

## Version control

The repo uses `.gitattributes` to keep line endings from creating noise:
pandas writes CRLF on Windows, while the repository stores LF. Re-running the
entire pipeline leaves all four CSVs showing **no diff**.

Two things worth knowing:

- `.gitattributes` deliberately uses plain `text`, not `text eol=lf`. Forcing
  LF in the working tree makes every pandas-written CSV disagree with its
  checked-out form, and `git status` then reports all four data files as
  modified after any rerun even though content is identical. This was tested,
  not assumed.
- `reports/Wayfair_Sales_Performance_Report.docx` **will** show as modified
  after every regeneration. A .docx is a zip archive with embedded
  timestamps, so its bytes change even when the document doesn't. Before
  committing one, check whether the content actually changed rather than
  assuming a diff is meaningful; if it hasn't, `git checkout --` the file.

## Verified environment

The pipeline was re-run end to end and is reproducible on library versions
well ahead of the minimums in `requirements.txt`:

| Package | Pinned minimum | Verified working on |
|---|---|---|
| pandas | >=2.0 | 3.0.1 |
| numpy | >=1.24 | 2.4.3 |
| Faker | >=20.0 | 40.37.0 |
| beautifulsoup4 | >=4.12 | 4.14.3 |
| matplotlib | >=3.7 | 3.10.8 |
| python-docx | >=1.1 | 1.2.0 |

With `RANDOM_SEED = 42`, `simulator.py` regenerates `raw_sales_data.csv`
byte-identically on Faker 40 despite being written against Faker 20, and
`rfm_analysis.py` reproduces `rfm_segments.csv` exactly on pandas 3.x. The
scraper still extracts 49/49 products, so the fragile CSS classes noted
above are still valid as of this run.
