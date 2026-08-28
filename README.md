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

# optional analysis tool (not part of the linear pipeline)
python scripts/scenario_simulator.py --list
python scripts/churn_model.py
python scripts/cohort_analysis.py
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
│   ├── customer_truth.csv        # simulator.py output (latent traits, ground truth)
│   ├── churn_predictions.csv     # churn_model.py output (per-customer risk)
│   ├── cohort_retention.csv      # cohort_analysis.py output (wide matrix)
│   ├── cohort_retention_long.csv # cohort_analysis.py output (tidy, for Power BI)
│   └── rfm_segments.csv          # rfm_analysis.py output (customer segments)
├── scripts/
│   ├── scraper.py
│   ├── clean_data.py
│   ├── simulator.py
│   ├── rfm_analysis.py
│   ├── generate_report.py
│   ├── scenario_simulator.py
│   ├── churn_model.py
│   └── cohort_analysis.py
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
- Generates a simulated transaction history against the real catalog, over a
  365-day window ending 2026-06-01.
- **Customers are not interchangeable.** Each draws two latent traits --
  a purchase rate (`lambda`, Gamma-distributed) and a dropout hazard giving a
  finite lifetime -- then buys via a Poisson process from their acquisition
  date until they churn or the window closes. This is the Pareto/NBD structure
  (Schmittlein, Morrison & Colombo 1987) used in non-contractual CLV work.
- **Acquisition dates are staggered** across the window rather than every
  customer existing from day one, so cohort analysis sees real monthly cohorts.
- **Transaction volume is an output, not an input.** Tune `NUM_CUSTOMERS` and
  the rate parameters; the ledger length follows. There is no
  `NUM_TRANSACTIONS` knob any more.
- Writes `data/customer_truth.csv` alongside the ledger, holding each
  customer's true `lambda`, lifetime, and churn status. That is ground truth
  for validating a churn/propensity model -- you can check whether the model
  recovers the parameters that generated the data.
- ±5% random price variance per transaction; `RANDOM_SEED = 42` for
  reproducibility; checks the catalog's `Source` column before proceeding.

#### Why the behavioral model exists

An earlier version assigned every transaction to a uniformly random customer
(`customer = random.choice(customers)`). That made each customer's future
purchases statistically independent of their past ones, which was invisible in
the descriptive outputs but broke everything predictive built on top:

| Symptom | Uniform random | Behavioral model |
|---|---|---|
| Churn model AUC (logistic regression) | 0.480 | **0.763** |
| `frequency` correlation with repurchase | -0.0003 | **+0.163** |
| Month-1 cohort retention | 35% (flat, no decay) | **10.6% (decaying)** |
| Cohort sizes across 12 months | 157 → 2 (artifact) | **115–138 (even)** |
| Repeat-purchase rate | 96.2% (implausible) | **28.0%** |

All five traced back to that one line. Note the sign on `frequency`: under the
old model, how often a customer had bought carried no information about whether
they would buy again, which is why any model scored at chance.

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

### `scenario_simulator.py`
A "what-if" tool for testing retention/discount campaigns against the RFM
segments before committing budget. Not part of the linear pipeline -- it reads
`rfm_segments.csv` and `raw_sales_data.csv` and answers questions like *"if we
give 15% off to At Risk customers and assume a 10% conversion lift, does it pay
for itself?"*

```bash
python scripts/scenario_simulator.py --list
python scripts/scenario_simulator.py --segment "At Risk / Can't Lose Them" --discount 15 --lift 10
python scripts/scenario_simulator.py --segment Champions --sweep --chart
```

Two things it does deliberately:

- **Accounts for discount cannibalization.** A discount is paid to everyone who
  redeems, including customers who would have bought anyway. Ignoring this is
  the most common way campaign ROI gets overstated. In the 15%/10% example
  above, $1,523 of discount goes to baseline buyers versus $152 to incremental
  ones -- which is why that scenario returns **-67% ROI** despite sounding
  reasonable.
- **Solves for break-even lift.** Rather than reporting ROI for one guessed
  lift, it reports the lift the campaign would need to break even, which is the
  number that actually drives a go/no-go call. At 5% discount that's a 10.4%
  lift; at 25% discount it's 81%.

**Baselines are measured out of sample.** Segments are rebuilt on history only,
then conversion is measured in a held-out 30-day window. Measuring conversion on
the same window that defined `recency` would be circular -- "At Risk" means "has
not purchased recently", so it reports a 0% baseline by construction and makes
every campaign look infinitely profitable. (Measured naively, At Risk shows 0.0%
and Champions 97.3%; measured properly, both sit near 38%.)

**Baselines now carry signal.** Since the simulator gained a behavioral model,
forward conversion ranges from 16.9% (Champions) down to 1.2% (Hibernating /
Lost) -- a 14x spread, correctly ordered. Under the old uniform-random
simulator every segment sat near 38% and `Hibernating / Lost` outranked
`Champions`, which was meaningless. The tool checks for both failure modes at
runtime (narrow spread, or Champions converting no better than Hibernating) and
prints a warning rather than trusting the numbers blindly. Rates are still
measured on simulated transactions, so pass `--baseline` with your own estimate
for a real decision; the arithmetic does not depend on the data having
predictive signal, since lift and baseline are inputs rather than estimates.

### `churn_model.py`
Scores each customer's probability of purchasing again in the next 30 days --
the forward-looking counterpart to `rfm_analysis.py`, which only describes what
already happened.

```bash
python scripts/churn_model.py
python scripts/cohort_analysis.py
```

**Validation.** Features come strictly from transactions on or before a cutoff;
the label is whether the customer purchased after it, so the windows never
overlap. Two numbers are reported because they answer different questions:

| | ROC AUC | Avg precision | Top-decile capture |
|---|---|---|---|
| Cross-validated (5-fold) | 0.768 | 0.146 | 30% |
| **Forward backtest** | **0.695** | **0.112** | **27%** |
| Random baseline | 0.500 | 0.053 | 10% |

The backtest fits at an earlier cutoff and scores at a later one, so training
data strictly precedes test data. **That is the number to quote** -- it is the
honest estimate of deployed behavior, and it is meaningfully lower than the
cross-validated figure, as it should be. Top-decile capture of 27% is a 2.7x
lift: contacting the riskiest 10% of customers reaches 27% of everyone who
would have bought.

**Validated against ground truth.** `customer_truth.csv` holds the latent
parameters `simulator.py` used, so the model can be checked against the process
that generated the data rather than only a held-out split:

| | Top decile | Bottom decile |
|---|---|---|
| True purchase rate (per year) | 2.92 | 1.07 |
| Actually churned in window | 36% | 97% |

Predicted probability correlates with the true purchase rate at rho = +0.296
and identifies genuinely-active customers at AUC 0.772.

**No `class_weight="balanced"`, deliberately.** Despite the 5% positive rate,
rebalancing rescales probabilities to a 50/50 world and destroys calibration:
mean predicted probability was 0.389 against an actual 0.053, with a Brier
score 4x worse (0.198 vs 0.049). Ranking did not improve either (AUC 0.761
balanced vs 0.768 unweighted). Since `scenario_simulator.py` consumes these as
expected values rather than only as a ranking, calibration is the property that
matters. The reflexive fix for imbalance cost accuracy here for nothing.

**Coefficients are partial effects, not standalone predictors.** The features
are heavily collinear (`monetary` ~ `frequency` x `aov`; `overdue_ratio` is
built from `recency`; `recency`/`tenure` correlate at 0.88). Under collinearity
a fitted coefficient can take the opposite sign to the feature's own
relationship with the outcome -- `monetary` correlates +0.024 with repurchase
but fits at -0.279. The script prints each coefficient beside that feature's
own correlation and flags the disagreements, so the chart cannot be read as
causal. Ranking quality is unaffected; only interpretation is.

### `cohort_analysis.py`
Groups customers by acquisition month and tracks what share of each cohort is
still buying in the months after. Answers what RFM cannot: not "who is valuable
now" but "how fast do customers decay, and is acquisition quality changing".

```bash
python scripts/cohort_analysis.py
```

Retention decays cleanly, and cohorts are evenly sized (101-138):

| Month | +1 | +2 | +3 | +4 | +5 | +6 |
|---|---|---|---|---|---|---|
| Avg retention | 10.6% | 8.5% | 7.1% | 5.6% | 4.7% | 3.2% |
| Cohorts averaged | 11 | 10 | 9 | 8 | 7 | 6 |

Four things the script does deliberately, each guarding a way cohort analysis
usually goes wrong:

- **Censoring is made explicit.** The matrix is triangular because later cohorts
  have had less time. Averaging down a column silently changes the sample, so
  the cohort count is printed beside every average -- a "month +8" figure drawn
  from three cohorts is not comparable to "month +1" drawn from eleven.
- **Cohort quality is compared at a fixed offset,** which is the only
  apples-to-apples comparison. The month +1 figures trend +0.44 pp per cohort,
  which looks like improving acquisition until you test it: **p = 0.086, not
  significant**. The script says so plainly. This dataset draws every customer
  from the same distribution, so flat is the correct answer, and a tool that
  reported a trend here would be wrong.
- **Blank means "not yet observed", never "nobody bought".** `groupby` drops
  both cases identically, so an observed month where a cohort went quiet would
  otherwise render exactly like a month that has not happened yet. Observed
  zeros are shown as `0%`; only genuinely future cells are blank.
- **Cohorts under 30 customers are flagged and excluded** from summary stats. A
  cohort of three produces cells reading 33% or 67% that look like findings on
  a heatmap and are one person's coin flip. (Under the old uniform-random
  simulator, later cohorts really did collapse to 2-4 customers.)

Also reports **cumulative revenue per cohort member**, the payback view that
retention percentages hide: by month +3 a cohort has delivered ~27% more per
member than at acquisition.

`data/cohort_retention_long.csv` is tidy long-format -- one row per
cohort/offset with retention, revenue per member, and cumulative revenue --
which is the shape Power BI's matrix visual wants.

## Known data characteristics (from the last full run)

- 49 real products, 1,500 simulated customers, 2,169 simulated transactions,
  $708,998.19 total simulated revenue.
- Revenue is still concentrated: the single highest-earning rug accounts for
  ~33% of revenue and the top 5 products for ~52%. That comes from the real
  catalog's price spread, not from the simulation.
- 28.0% of customers placed 2+ orders, and 58.0% churned before the window
  closed. Both are properties of the behavioral model rather than claims about
  real Wayfair shoppers, but they are at least in a plausible range for
  big-ticket home goods.
- Products are still chosen uniformly at random per transaction. Real customers
  favor cheaper items, so the revenue concentration above is if anything
  overstated. Tackling that would be the next realism improvement.

## Report

`reports/Wayfair_Sales_Performance_Report.docx` — formal Word report,
generated by `scripts/generate_report.py`. Covers KPIs, monthly revenue
trend, top/bottom products, revenue concentration (Pareto), customer
purchase frequency, and **RFM customer segmentation**. States data
provenance (real catalog vs. simulated transactions) up front, and repeats
the simulation caveat inside the segmentation section so segment sizes
aren't read as real-world customer distribution.

Segment breakdown from the last run (sums to the $708,998.19 revenue total):

| Segment | Customers | % of base | Revenue | % of revenue |
|---|---|---|---|---|
| Champions | 186 | 12.4% | $219,214.45 | 30.9% |
| Loyal Customers | 230 | 15.3% | $139,850.04 | 19.7% |
| Promising / New | 322 | 21.5% | $117,082.57 | 16.5% |
| About to Sleep / Average | 162 | 10.8% | $14,926.18 | 2.1% |
| At Risk / Can't Lose Them | 322 | 21.5% | $129,793.64 | 18.3% |
| Hibernating / Lost | 278 | 18.5% | $88,131.31 | 12.4% |

## Natural next steps

- Feed `churn_predictions.csv` into `scenario_simulator.py` so campaigns can be
  targeted by predicted risk rather than by segment membership. The model is
  calibrated, so its probabilities can be used as expected values directly.
- Weight product selection by price so cheaper items sell more often. Products
  are currently drawn uniformly, which overstates revenue concentration.
- If the scraper ever reports 0 products, re-derive the CSS classes (see
  fragility note above) before assuming the page has no products.

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
