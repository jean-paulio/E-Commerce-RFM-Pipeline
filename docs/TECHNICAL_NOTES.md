# Technical Notes

Engineering reference for the pipeline: design decisions, known fragilities,
verification results, and the reasoning behind non-obvious choices. For a
project overview, see the [README](../README.md).

This document exists so a fresh session (human or AI) can pick up the project
without re-deriving decisions that were already made and tested.

## Quick start

```bash
pip install -r requirements.txt

python scripts/scraper.py        # data/wayfair_page.html -> data/product_catalog.csv
python scripts/clean_data.py     # data/product_catalog.csv -> data/product_catalog_clean.csv
python scripts/simulator.py      # data/product_catalog_clean.csv -> data/raw_sales_data.csv
python scripts/rfm_analysis.py   # data/raw_sales_data.csv -> data/rfm_segments.csv

# optional analysis tools (not part of the linear pipeline)
python scripts/scenario_simulator.py --list
python scripts/churn_model.py
python scripts/cohort_analysis.py
```

All seven scripts resolve paths relative to their own file location
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
│   ├── scenario_simulator.py
│   ├── churn_model.py
│   └── cohort_analysis.py
├── reports/
│   └── charts/                   # PNGs written by the analysis scripts (gitignored)
├── docs/
│   └── TECHNICAL_NOTES.md        # this file
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
- **Customers are not interchangeable.** Each draws a purchase rate
  (`lambda`, Gamma-distributed) and a dropout hazard (`mu`, Gamma-distributed),
  then an exponential lifetime at that hazard, and buys via a Poisson process
  from acquisition until they churn or the window closes. This is the
  Pareto/NBD structure (Schmittlein, Morrison & Colombo 1987): the Gamma-mixed
  Poisson gives NBD purchase counts, and the Gamma-mixed exponential lifetime
  is Lomax (Pareto type II), which is where the "Pareto" comes from.
- **The dropout process was wrong until 2026-09-17.** The code drew
  `lifetime_days` straight from a Gamma, which is a reasonable behavioral model
  but is not Pareto/NBD -- and the module docstring described a hazard the code
  never drew. Drawing the hazard first gives lifetimes a much heavier tail:
  median 79 days against a 162-day mean.
- `DROPOUT_SHAPE = 2.5` with `LIFETIME_MEAN_DAYS = 165` was chosen so the
  corrected process still yields a plausible dataset (23% repeat rate, 66%
  churned inside the window). Shape values below ~2 push the repeat rate under
  21%.
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
| Churn model AUC (6-feature probe, logistic regression) | 0.480 | **0.739** |
| `frequency` correlation with repurchase | -0.0003 | **+0.126** |
| Month-1 cohort retention | 35% (flat, no decay) | **9.0% (decaying)** |
| Cohort sizes across 12 months | 157 → 2 (artifact) | **109–155 (even)** |
| Repeat-purchase rate | 96.2% (implausible) | **23.0%** |

All five traced back to that one line. Note the sign on `frequency`: under the
old model, how often a customer had bought carried no information about whether
they would buy again, which is why any model scored at chance.

### `rfm_analysis.py`
- Computes Recency / Frequency / Monetary per customer from
  `raw_sales_data.csv`, scores each 1–5, and assigns one of six segment
  labels.
- **Recency and monetary use tie-aware quintiles**: percentile rank with
  `method="average"`, so equal values always land on the same score. Buckets
  are uneven whenever values repeat, which is the correct trade.
- **Frequency uses explicit order-count bands** (1 / 2 / 3 / 4–5 / 6+). Its
  distribution is far too concentrated for quintiles: 77% of customers place
  exactly one order.
- **⚠️ Do not reintroduce `.rank(method="first")` before `qcut`.** It was here
  until 2026-09-17 as a way to stop `qcut` raising "duplicate bin edge" on tied
  values. It works, but it breaks ties by row order, so identical customers get
  different scores: the one-order customers were spread across F-scores 1–4,
  putting 18 one-time buyers in Champions and 109 in Loyal Customers. Passing
  `duplicates='drop'` to `qcut` is not the alternative either -- it raises a
  different error when explicit `labels` are supplied. Score ties explicitly
  instead.
- **Segment rules test frequency first.** With 77% of customers at one order,
  the classic recency-first grid put 91% of customers into two labels and left
  `About to Sleep / Average` empty. Repeat buyers (3+ orders) split by recency
  and value; everyone else splits by recency alone.
- `describe()` warns if any one-order customer lands in a segment that implies
  repeat buying -- the original symptom of the tie bug, now a standing check.
- The scoring functions are imported by `scenario_simulator.py` rather than
  copied. The copy that used to live there would have kept the old tie
  behavior after this file was fixed.
- `snapshot_date` is derived from the data itself
  (`max(InvoiceDate) + 1 day`), not hardcoded, so it stays correct even if
  `simulator.py`'s date window changes later.

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
  the most common way campaign ROI gets overstated. Offering 15% off to the
  521 customers in `Promising / New` and assuming a 10% lift sends $1,093 of
  discount to baseline buyers versus $109 to incremental ones -- which is why
  that scenario returns **-70% ROI** despite sounding reasonable.
- **Solves for break-even lift.** Rather than reporting ROI for one guessed
  lift, it reports the lift the campaign would need to break even, which is the
  number that actually drives a go/no-go call. For `Promising / New`, a 5%
  discount needs a 14.6% lift to break even; a 25% discount needs 88.0%, close
  to doubling the segment's conversion rate just to get the money back.

**Baselines are measured out of sample.** Segments are rebuilt on history only,
then conversion is measured in a held-out 30-day window. Measuring conversion on
the same window that defined `recency` would be circular -- a lapsed segment
means "has not purchased recently", so it reports a 0% baseline by construction
and makes every campaign look infinitely profitable. (Measured naively on the
current data, every lapsed segment shows 0.0% and Champions 36.8%; measured
properly out of sample, Hibernating / Lost is 1.1% and Champions 17.9%.)

**Baselines carry signal, and unmeasurable segments say so.** Forward conversion
ranges from 17.9% (Champions) down to 1.1% (Hibernating / Lost), correctly
ordered. `At Risk / Can't Lose Them` holds only 22 customers with no observed
conversions, so no rate can be estimated for it: a zero denominator would make
the max/min spread infinite and sail past any "is there signal here?" check,
which is exactly what happened when the corrected segmentation first shrank that
segment. Segments below `MIN_SEGMENT_OBSERVATIONS` or with no conversions are
now reported as unmeasured, and the spread is computed from the rest. Rates are
still measured on simulated transactions, so pass `--baseline` with your own
estimate for a real decision; the arithmetic does not depend on the data having
predictive signal, since lift and baseline are inputs rather than estimates.

### `churn_model.py`
Scores each customer's probability of purchasing again in the next 30 days --
the forward-looking counterpart to `rfm_analysis.py`, which only describes what
already happened.

```bash
python scripts/churn_model.py
```

**Validation.** Features come strictly from transactions on or before a cutoff;
the label is whether the customer purchased after it, so the windows never
overlap. Two numbers are reported because they answer different questions:

| | ROC AUC | Avg precision | Top-decile capture |
|---|---|---|---|
| Selection snapshot (5-fold CV) | 0.775 | 0.118 | 31% |
| **Forward backtest** | **0.696** | **0.096** | **31%** |
| Random baseline | 0.500 | 0.039 | 10% |

The backtest fits at an earlier cutoff and scores at a later one, so training
data strictly precedes test data. **That is the number to quote** -- it is the
honest estimate of deployed behavior, and it is meaningfully lower than the
cross-validated figure, as it should be. Top-decile capture of 31% is a 3.1x
lift: contacting the riskiest 10% of customers reaches 31% of everyone who
would have bought.

**Model selection uses the earlier snapshot only.** Until 2026-09-17 all three
candidates were scored on the test period and the winner picked from those
results, which let the test period influence the choice -- the same leak as
tuning on a test set. This is not academic: selecting honestly on the earlier
snapshot picks random forest, while the old procedure picked logistic
regression. Their forward AUCs are within 0.01 of each other, so the headline
number barely moved, but the procedure has to hold regardless of how it falls.

**Scores are out-of-time.** `churn_predictions.csv` and the ground-truth check
use the model fitted on the earlier snapshot. Fitting on the test snapshot and
scoring it -- which is what this did before -- makes both in-sample.

**Validated against ground truth.** `customer_truth.csv` holds the latent
parameters `simulator.py` used, so the model can be checked against the process
that generated the data rather than only a held-out split:

| | Top decile | Bottom decile |
|---|---|---|
| True purchase rate (per year) | 2.69 | 1.30 |
| Actually churned in window | 42% | 86% |

Predicted probability correlates with the true purchase rate at rho = +0.174
and identifies genuinely-active customers at AUC 0.692. These are lower than
the in-sample figures this section used to quote (+0.296 and 0.772), because
the scores are now produced by a model that never saw these outcomes.

`churn_predictions.csv` includes a `risk_decile` column where **10 = highest
churn risk**. It was originally ranked on repurchase probability, which
silently inverted it so that decile 1 was the riskiest -- the opposite of what
anyone sorting a dashboard by risk would assume.

**No `class_weight="balanced"`, deliberately.** Despite the ~4% positive rate,
rebalancing rescales probabilities to a 50/50 world: mean predicted probability
was 0.394 against an actual 0.039, with a Brier score 5x worse (0.2010 vs
0.0376). Ranking did not improve either (AUC 0.724 balanced vs 0.728
unweighted), so the reflexive fix for imbalance cost accuracy for nothing.

**But the probabilities are not established as calibrated.** Dropping the class
weighting is necessary, not sufficient. Out of time the model scores a Brier of
0.0373 against 0.0378 for predicting the base rate for every customer -- a
skill score of +1.2%, which is parity. The script prints this comparison on
every run and says so in words. Earlier versions of these notes claimed
`scenario_simulator.py` consumed the probabilities as expected values; it never
did, and on this evidence it should not.

**Coefficients are partial effects, not standalone predictors.** The features
are collinear (`monetary` ~ `frequency` x `aov` at 0.84; `recency`/`tenure` at
0.89; `frequency`/`units` at 0.72). Under collinearity
a fitted coefficient can take the opposite sign to the feature's own
relationship with the outcome -- in the current run `units` correlates +0.101
with repurchase but fits at -0.097. The script prints each coefficient beside that feature's
own correlation and flags the disagreements, so the chart cannot be read as
causal. Ranking quality is unaffected; only interpretation is.

### `cohort_analysis.py`
Groups customers by acquisition month and tracks what share of each cohort is
still buying in the months after. Answers what RFM cannot: not "who is valuable
now" but "how fast do customers decay, and is acquisition quality changing".

```bash
python scripts/cohort_analysis.py
```

Retention decays cleanly, and cohorts are evenly sized (109-155):

| Month | +1 | +2 | +3 | +4 | +5 | +6 |
|---|---|---|---|---|---|---|
| Avg retention | 9.0% | 6.9% | 5.6% | 5.8% | 3.9% | 3.4% |
| Cohorts averaged | 11 | 10 | 9 | 8 | 7 | 6 |

Four things the script does deliberately, each guarding a way cohort analysis
usually goes wrong:

- **Censoring is made explicit.** The matrix is triangular because later cohorts
  have had less time. Averaging down a column silently changes the sample, so
  the cohort count is printed beside every average -- a "month +8" figure drawn
  from three cohorts is not comparable to "month +1" drawn from eleven.
- **Cohort quality is compared at a fixed offset,** which is the only
  apples-to-apples comparison. The month +1 figures trend -0.37 pp per cohort,
  which looks like declining acquisition until you test it: **p = 0.075, not
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

- 49 real products, 1,500 simulated customers, 2,077 simulated transactions,
  $700,701.67 total simulated revenue.
- Revenue is still concentrated: the single highest-earning rug accounts for
  ~23% of revenue and the top 5 products for ~51%. That comes from the real
  catalog's price spread, not from the simulation.
- 23.0% of customers placed 2+ orders (77.0% ordered exactly once), and 66.3%
  churned before the window closed. Both are properties of the behavioral model rather than claims about
  real Wayfair shoppers, but they are at least in a plausible range for
  big-ticket home goods.
- Products are still chosen uniformly at random per transaction. Real customers
  favor cheaper items, so the revenue concentration above is if anything
  overstated. Tackling that would be the next realism improvement.

## Report

The performance report is authored by hand, not generated. `generate_report.py`
was removed on 2026-09-17 so the report and its visuals could be produced
directly; the generator remains in git history if it is ever wanted back.

`reports/Wayfair_Sales_Performance_Report.docx` is the last generated version,
kept for reference until a hand-authored report replaces it. Treat it as a
frozen snapshot: nothing regenerates it, so it will not follow later changes to
the data. In particular, its RFM segmentation section is built from
`rfm_segments.csv` and will go stale as soon as the frequency scoring is
corrected.

Anything the report states about customer behavior should carry the same
provenance caveat the pipeline uses elsewhere: the product catalog is real,
while every customer, order, and date is simulated.

Segment breakdown from the last run (sums to the $700,701.67 revenue total):

| Segment | Customers | % of base | Revenue | % of revenue |
|---|---|---|---|---|
| Hibernating / Lost | 579 | 38.6% | $223,703.14 | 31.9% |
| Promising / New | 521 | 34.7% | $199,624.07 | 28.5% |
| Champions | 76 | 5.1% | $117,623.75 | 16.8% |
| About to Sleep / Average | 270 | 18.0% | $92,592.30 | 13.2% |
| Loyal Customers | 32 | 2.1% | $40,254.56 | 5.7% |
| At Risk / Can't Lose Them | 22 | 1.5% | $26,903.85 | 3.8% |

## Natural next steps

- Feed `churn_predictions.csv` into `scenario_simulator.py` so campaigns can be
  targeted by predicted risk rather than by segment membership. Use the scores
  as a ranking (who to contact), not as expected values -- the forward Brier
  skill is only +1.2%, so they would need recalibrating first.
- Weight product selection by price so cheaper items sell more often. Products
  are currently drawn uniformly, which overstates revenue concentration.
- If the scraper ever reports 0 products, re-derive the CSS classes (see
  fragility note above) before assuming the page has no products.

## Version control

The repo uses `.gitattributes` to keep line endings from creating noise:
pandas writes CRLF on Windows, while the repository stores LF. Re-running the
entire pipeline leaves all four CSVs showing **no diff**.

One thing worth knowing:

- `.gitattributes` deliberately uses plain `text`, not `text eol=lf`. Forcing
  LF in the working tree makes every pandas-written CSV disagree with its
  checked-out form, and `git status` then reports all four data files as
  modified after any rerun even though content is identical. This was tested,
  not assumed.

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
| scikit-learn | >=1.3 | 1.9.0 |
| scipy | >=1.10 | 1.18.1 |

With `RANDOM_SEED = 42` the pipeline is fully reproducible: a clean run from
`wayfair_page.html` regenerates `raw_sales_data.csv`, `customer_truth.csv`, and
`rfm_segments.csv` identically (verified after the behavioral-model rewrite). The
scraper still extracts 49/49 products, so the fragile CSS classes noted
above are still valid as of this run.
