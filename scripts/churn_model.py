"""
churn_model.py

Predicts each customer's probability of purchasing again in the next 30 days
-- equivalently, their churn risk -- from behavior observed before a cutoff
date.

This is the forward-looking counterpart to rfm_analysis.py. RFM describes what
customers already did; this scores what they are likely to do next, which is
what a retention budget actually needs.

METHOD
------
Features are built strictly from transactions on or before a cutoff, and the
label is whether the customer purchased in the window after it. Feature and
label windows never overlap, so there is no leakage.

Two evaluations are reported, because they answer different questions:

  * Cross-validated (5-fold, stratified) at a single cutoff -- how well the
    model separates customers given this much data.
  * A forward backtest -- fit at an earlier cutoff, scored at a later one, so
    training data strictly precedes test data. This is the honest estimate of
    how the model behaves deployed, and it is the number to quote.

With a ~5% positive rate, ROC AUC alone flatters a model. Average precision
(against the base rate) and top-decile capture are reported alongside it, and
calibration is checked because the scenario simulator consumes these
probabilities as expected values, not just rankings.

GROUND TRUTH
------------
data/customer_truth.csv holds the latent parameters simulator.py used to
generate each customer. That allows a check most projects cannot make: whether
predicted risk tracks the true purchase rate and true churn state, rather than
only agreeing with a held-out split of the same data.

Outputs:
    data/churn_predictions.csv     per-customer risk scores
    reports/charts/churn_*.png     ROC, calibration, gains, feature weights
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CHART_DIR = os.path.join(SCRIPT_DIR, "..", "reports", "charts")

SALES_FILE = os.path.join(DATA_DIR, "raw_sales_data.csv")
TRUTH_FILE = os.path.join(DATA_DIR, "customer_truth.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "churn_predictions.csv")

HORIZON_DAYS = 30
RANDOM_STATE = 42

NAVY = "#1F3864"
GOLD = "#C9973E"
BODY_GRAY = "#444444"

FEATURES = [
    "recency", "frequency", "monetary", "aov", "tenure", "units",
    "mean_gap", "overdue_ratio",
]


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------

def build_features(sales: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """RFM-style features from transactions at or before `cutoff`.

    Only customers who exist by the cutoff are included -- a customer whose
    first purchase comes later has no history to score.
    """
    hist = sales[sales["InvoiceDate"] <= cutoff]
    if hist.empty:
        raise ValueError(f"no transactions at or before {cutoff:%Y-%m-%d}")

    g = hist.groupby("CustomerID")
    orders = g["InvoiceNo"].nunique()
    spend = g["Revenue"].sum()

    feats = pd.DataFrame({
        "recency": (cutoff - g["InvoiceDate"].max()).dt.days,
        "frequency": orders,
        "monetary": spend,
        "aov": spend / orders,
        "tenure": (cutoff - g["InvoiceDate"].min()).dt.days,
        "units": g["Quantity"].sum(),
    })

    # Average gap between purchases, and how far past it the customer is.
    # "Overdue" is usually the single strongest signal: a customer who buys
    # every 40 days and last bought 200 days ago is behaving differently from
    # one who buys every 300 days and last bought 200 days ago, even though
    # their recency is identical.
    feats["mean_gap"] = np.where(
        feats["frequency"] > 1,
        feats["tenure"] / (feats["frequency"] - 1).clip(lower=1),
        np.nan,
    )
    typical_gap = feats["mean_gap"].median()
    feats["mean_gap"] = feats["mean_gap"].fillna(typical_gap)
    feats["overdue_ratio"] = feats["recency"] / feats["mean_gap"].clip(lower=1)
    return feats


def build_labels(sales: pd.DataFrame, cutoff: pd.Timestamp,
                 horizon: int = HORIZON_DAYS) -> set:
    window = sales[(sales["InvoiceDate"] > cutoff)
                   & (sales["InvoiceDate"] <= cutoff + pd.Timedelta(days=horizon))]
    return set(window["CustomerID"])


def dataset_at(sales: pd.DataFrame, cutoff: pd.Timestamp, horizon=HORIZON_DAYS):
    X = build_features(sales, cutoff)
    buyers = build_labels(sales, cutoff, horizon)
    y = X.index.isin(buyers).astype(int)
    return X[FEATURES], y


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def candidate_models():
    """Candidate classifiers.

    Deliberately no `class_weight="balanced"`, despite the 5% positive rate.
    Rebalancing rescales predicted probabilities to a 50/50 world, which
    destroys calibration -- with balanced weights the mean predicted
    probability was 0.389 against an actual rate of 0.053, and the Brier score
    was 4x worse (0.198 vs 0.049). Ranking did not improve either: AUC was
    0.761 balanced vs 0.768 unweighted. Since these probabilities feed
    scenario_simulator.py as expected values rather than only as a ranking,
    calibration is the property that matters, and the usual reflex for
    imbalance costs it for nothing here.
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=8,
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE),
    }


def capture_at(y_true, scores, top_frac):
    """Share of actual buyers found in the top `top_frac` of ranked customers.

    The metric a retention team actually plans against: "if we can only
    contact 10% of the base, how many of the people who would have bought do
    we reach?"
    """
    n = max(1, int(round(len(scores) * top_frac)))
    idx = np.argsort(scores)[::-1][:n]
    total = y_true.sum()
    return y_true[idx].sum() / total if total else float("nan")


def evaluate_cv(X, y, model, folds=5):
    cv = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "auc": roc_auc_score(y, proba),
        "ap": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "capture10": capture_at(y, proba, 0.10),
        "capture20": capture_at(y, proba, 0.20),
        "proba": proba,
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_header(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def report_cv(X, y, results, base_rate):
    print_header("MODEL COMPARISON (5-fold cross-validated)")
    print(f"  positives: {y.sum()} of {len(y)} customers "
          f"({base_rate*100:.1f}% base rate)\n")
    print(f"  {'model':<22}{'ROC AUC':>9}{'avg prec':>10}{'Brier':>9}"
          f"{'top10%':>9}{'top20%':>9}")
    for name, r in results.items():
        print(f"  {name:<22}{r['auc']:>9.3f}{r['ap']:>10.3f}{r['brier']:>9.4f}"
              f"{r['capture10']*100:>8.0f}%{r['capture20']*100:>8.0f}%")
    print(f"\n  random baseline        {0.5:>9.3f}{base_rate:>10.3f}"
          f"{'':>9}{10:>8.0f}%{20:>8.0f}%")
    print("  (avg precision is only meaningful against the base rate;"
          f" {base_rate:.3f} = chance)")


def report_backtest(name, metrics, base_rate):
    print_header(f"FORWARD BACKTEST -- {name}")
    print("  trained on an earlier cutoff, scored on a later one, so all")
    print("  training data strictly precedes all test data.\n")
    print(f"  ROC AUC            {metrics['auc']:.3f}")
    print(f"  Average precision  {metrics['ap']:.3f}   "
          f"(base rate {base_rate:.3f} = chance)")
    print(f"  Brier score        {metrics['brier']:.4f}   (lower is better)")
    print(f"  Top-decile capture {metrics['capture10']*100:.0f}%   "
          f"of buyers found in the top 10% of scores")
    lift = metrics["capture10"] / 0.10
    print(f"  Lift over random   {lift:.1f}x")


def report_coefficients(model, X, y):
    """Coefficients alongside each feature's own correlation with the label.

    Showing both matters because these features are heavily collinear
    (monetary is roughly frequency x aov; overdue_ratio is built from
    recency). Under collinearity a fitted coefficient can take the opposite
    sign to the feature's actual relationship with the outcome -- it is
    reporting a partial effect holding correlated features fixed, not "this
    feature predicts churn". Flagging the disagreements keeps the chart from
    being read as causal.
    """
    lr = model.named_steps["logisticregression"]
    coefs = pd.Series(lr.coef_[0], index=X.columns)
    univariate = pd.Series(
        {c: np.corrcoef(X[c], y)[0, 1] for c in X.columns})
    coefs = coefs.sort_values(key=abs, ascending=False)

    print_header("WHAT THE MODEL LEARNED (standardized coefficients)")
    print("  positive = raises repurchase probability")
    print(f"  {'feature':<16}{'coef':>8}{'own corr':>10}\n")

    conflicts = []
    for feat, val in coefs.items():
        uni = univariate[feat]
        clash = val * uni < 0 and abs(uni) > 0.02 and abs(val) > 0.05
        if clash:
            conflicts.append(feat)
        bar = "#" * int(min(abs(val) * 18, 26))
        flag = "  <- sign flip" if clash else ""
        print(f"  {feat:<16}{val:>+8.3f}{uni:>+10.3f}  {bar}{flag}")

    if conflicts:
        plural = len(conflicts) > 1
        verb = "carry coefficients" if plural else "carries a coefficient"
        whose = "their" if plural else "its"
        print(f"\n  NOTE: {', '.join(conflicts)} {verb} opposite in")
        print(f"  sign to {whose} own correlation with the outcome. Those features")
        print("  are collinear with others in the model, so read coefficients as")
        print("  partial effects, not as standalone predictors. Ranking quality")
        print("  (AUC, capture) is unaffected -- only interpretation is.")
    return coefs


def report_ground_truth(scores: pd.Series):
    """Check predictions against the parameters that generated the data."""
    if not os.path.exists(TRUTH_FILE):
        return
    truth = pd.read_csv(TRUTH_FILE).set_index("CustomerID")
    joined = truth.join(scores.rename("p_repurchase"), how="inner")
    joined = joined.dropna(subset=["p_repurchase"])
    if joined.empty:
        return

    print_header("VALIDATION AGAINST GROUND TRUTH")
    print("  customer_truth.csv holds the latent parameters simulator.py used,")
    print("  so predictions can be checked against reality, not just a split.")
    print("  Scores below are predicted REPURCHASE PROBABILITY, so higher")
    print("  should mean more active (churn risk is its complement).\n")

    r_lambda = joined["p_repurchase"].corr(joined["true_lambda_per_year"],
                                           method="spearman")
    still_active = (~joined["churned_before_window_end"]).astype(int)
    auc_active = (roc_auc_score(still_active, joined["p_repurchase"])
                  if 0 < still_active.sum() < len(still_active) else float("nan"))

    print(f"  P(repurchase) vs true purchase rate    rho = {r_lambda:+.3f}"
          f"   (want > 0)")
    print(f"  identifying genuinely-active customers AUC = {auc_active:.3f}"
          f"   (want > 0.5)")

    deciles = pd.qcut(joined["p_repurchase"].rank(method="first"), 10, labels=False)
    top, bottom = joined[deciles == 9], joined[deciles == 0]
    print(f"\n  {'':<34}{'top decile':>12}{'bottom decile':>15}")
    print(f"  {'true purchase rate (per year)':<34}"
          f"{top['true_lambda_per_year'].mean():>12.2f}"
          f"{bottom['true_lambda_per_year'].mean():>15.2f}")
    print(f"  {'actually churned in window':<34}"
          f"{top['churned_before_window_end'].mean()*100:>11.0f}%"
          f"{bottom['churned_before_window_end'].mean()*100:>14.0f}%")


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------

def save_charts(y, proba, coefs, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(CHART_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # ROC
    ax = axes[0][0]
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, color=NAVY, lw=2.4)
    ax.plot([0, 1], [0, 1], "--", color="#BBBBBB", lw=1.2)
    ax.set_title(f"ROC — AUC {roc_auc_score(y, proba):.3f}", color=NAVY,
                 fontweight="bold")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")

    # Cumulative gains
    ax = axes[0][1]
    order = np.argsort(proba)[::-1]
    gains = np.cumsum(y[order]) / y.sum()
    pct = np.arange(1, len(y) + 1) / len(y)
    ax.plot(pct * 100, gains * 100, color=NAVY, lw=2.4, label="model")
    ax.plot([0, 100], [0, 100], "--", color="#BBBBBB", lw=1.2, label="random")
    ax.axvline(10, color=GOLD, lw=1.4, ls=":")
    ax.set_title("Cumulative gains", color=NAVY, fontweight="bold")
    ax.set_xlabel("% of customers contacted (ranked by risk)")
    ax.set_ylabel("% of actual buyers reached")
    ax.legend(frameon=False, fontsize=9)

    # Calibration
    ax = axes[1][0]
    frac, mean_pred = calibration_curve(y, proba, n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac, "o-", color=NAVY, lw=2)
    ax.plot([0, max(mean_pred)], [0, max(mean_pred)], "--", color="#BBBBBB", lw=1.2)
    ax.set_title("Calibration", color=NAVY, fontweight="bold")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")

    # Coefficients
    ax = axes[1][1]
    c = coefs.sort_values()
    ax.barh(c.index, c.values,
            color=[GOLD if v < 0 else NAVY for v in c.values])
    ax.axvline(0, color="#888888", lw=1)
    ax.set_title("Standardized coefficients", color=NAVY, fontweight="bold")
    ax.tick_params(labelsize=9)

    for row in axes:
        for a in row:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.tick_params(colors=BODY_GRAY)

    fig.suptitle(f"Churn / next-purchase model — {name}", color=NAVY,
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "churn_model.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------

def main():
    if not os.path.exists(SALES_FILE):
        print(f"Error: {SALES_FILE} not found. Run simulator.py first.")
        sys.exit(1)

    sales = pd.read_csv(SALES_FILE, parse_dates=["InvoiceDate"])
    sales["Revenue"] = sales["UnitPrice"] * sales["Quantity"]

    last = sales["InvoiceDate"].max()
    cutoff_test = last - pd.Timedelta(days=HORIZON_DAYS)
    cutoff_train = cutoff_test - pd.Timedelta(days=HORIZON_DAYS)

    X, y = dataset_at(sales, cutoff_test)
    base_rate = y.mean()
    print(f"Scoring window   : {HORIZON_DAYS} days")
    print(f"Feature cutoff   : {cutoff_test:%Y-%m-%d}  "
          f"({len(X):,} customers with history)")
    print(f"Label window     : {cutoff_test + pd.Timedelta(days=1):%Y-%m-%d}"
          f" -> {last:%Y-%m-%d}")

    results = {name: evaluate_cv(X, y, m) for name, m in candidate_models().items()}
    report_cv(X, y, results, base_rate)

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  best by ROC AUC: {best_name}")

    # --- forward backtest -------------------------------------------------
    X_tr, y_tr = dataset_at(sales, cutoff_train)
    model = candidate_models()[best_name]
    model.fit(X_tr, y_tr)
    proba_bt = model.predict_proba(X)[:, 1]
    bt = {
        "auc": roc_auc_score(y, proba_bt),
        "ap": average_precision_score(y, proba_bt),
        "brier": brier_score_loss(y, proba_bt),
        "capture10": capture_at(y, proba_bt, 0.10),
    }
    report_backtest(best_name, bt, base_rate)

    # --- interpretability (always from logistic regression) ---------------
    lr = candidate_models()["Logistic Regression"]
    lr.fit(X, y)
    coefs = report_coefficients(lr, X, y)

    # --- final scores on all customers ------------------------------------
    final = candidate_models()[best_name]
    final.fit(X, y)
    scores = pd.Series(final.predict_proba(X)[:, 1], index=X.index)

    out = X.copy()
    out["repurchase_probability"] = scores.round(4)
    out["churn_risk"] = (1 - scores).round(4)
    # 10 = highest churn risk, so sorting descending puts the customers most in
    # need of attention first -- the direction a dashboard user will assume.
    # (Ranking on repurchase probability instead silently inverts it.)
    out["risk_decile"] = pd.qcut((1 - scores).rank(method="first"), 10,
                                 labels=False) + 1
    out["actually_purchased"] = y
    out.sort_values("churn_risk", ascending=False).to_csv(OUTPUT_FILE)

    report_ground_truth(scores)

    path = save_charts(y, results[best_name]["proba"], coefs, best_name)
    print(f"\nWrote {OUTPUT_FILE}")
    print(f"Charts -> {path}")


if __name__ == "__main__":
    main()
