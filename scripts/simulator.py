"""
simulator.py

Generates a simulated sales transaction history (raw_sales_data.csv) using
a REAL scraped product catalog and SIMULATED customers/transactions.

Customer and transaction data is synthetic by design, since real Wayfair
sales history is proprietary and not something we have (or should have)
access to. The product catalog, however, is expected to be real scraped
data -- this script checks for that and warns/aborts if the catalog looks
like fallback/sample data instead.

BEHAVIORAL MODEL
----------------
Customers are not interchangeable. Each one draws two latent traits:

    lambda  purchases per day while active   ~ Gamma
    mu      dropout hazard (per day)         ~ Gamma
    tau     lifetime                         ~ Exponential(mu)

and then buys via a Poisson process from their acquisition date until they
churn or the window closes. This is the Pareto/NBD structure (Schmittlein,
Morrison & Colombo 1987) that underpins most non-contractual CLV work: the
Gamma-mixed Poisson gives the NBD purchase counts, and the Gamma-mixed
exponential lifetime is Lomax (Pareto type II), which supplies the Pareto.

An earlier version of this script assigned every transaction to a uniformly
random customer:

    customer = random.choice(customers)     # old

That made a customer's future purchases statistically independent of their
past ones, with three consequences that only show up downstream:

  * churn/propensity models scored AUC ~0.50, because there was no signal
    to find -- not a modeling failure, an absence of structure in the data;
  * cohort retention curves came out flat at ~35% instead of decaying, since
    nobody ever churned;
  * the repeat-purchase rate sat at 96%, which is implausible for big-ticket
    home goods.

All three trace back to that one line. The model here fixes the cause rather
than the symptoms.

Acquisition dates are staggered across the window instead of every customer
existing from day one, so cohort analysis sees genuine monthly cohorts rather
than an artifact of when each customer's first random draw happened to land.

Transaction volume is an OUTPUT of this process, not an input. Tune
NUM_CUSTOMERS and the rate parameters; the ledger length follows.

Outputs:
    data/raw_sales_data.csv    the transaction ledger (schema unchanged)
    data/customer_truth.csv    each customer's latent traits -- ground truth
                               for validating churn/propensity models
"""

import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CATALOG_FILE = os.path.join(DATA_DIR, "product_catalog_clean.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_sales_data.csv")
TRUTH_FILE = os.path.join(DATA_DIR, "customer_truth.csv")

NUM_CUSTOMERS = 1500
# A full calendar year (2025-01-01 .. 2025-12-31). END_DATE is exclusive: the
# window runs back WINDOW_DAYS from it, and the last transaction lands on the
# day before. A calendar year gives clean quarters for reporting.
END_DATE = datetime(2026, 1, 1)
WINDOW_DAYS = 365
START_DATE = END_DATE - timedelta(days=WINDOW_DAYS)

# --- purchase rate while active -------------------------------------------
# Gamma with shape < 1 is strongly right-skewed: most customers buy rarely,
# a minority buy often. That spread is what makes frequency predictive.
LAMBDA_SHAPE = 0.9
LAMBDA_MEAN_PER_YEAR = 1.7          # mean repeat purchases/year while active

# --- dropout ---------------------------------------------------------------
# True Pareto/NBD dropout: each customer draws a dropout HAZARD from a Gamma,
# then an exponential lifetime at that hazard. The lifetime is therefore
# Lomax (Pareto type II) distributed -- the "Pareto" the model is named for.
#
# An earlier version drew the lifetime straight from a Gamma, which is a
# reasonable model but is not Pareto/NBD, and the docstring above claimed the
# hazard was drawn even though the code never drew one.
#
# DROPOUT_SHAPE is s in Gamma(s, beta); the scale is derived so that the mean
# lifetime equals LIFETIME_MEAN_DAYS, since E[1/mu] = 1 / (scale * (s - 1)).
DROPOUT_SHAPE = 2.5
LIFETIME_MEAN_DAYS = 165

QUANTITY_CHOICES = [1, 2, 5]
QUANTITY_WEIGHTS = [0.8, 0.15, 0.05]
PRICE_VARIANCE_PCT = 0.05  # +/- 5% to mimic minor price fluctuation over the year

RANDOM_SEED = 42  # set for reproducible simulated datasets; remove for fresh randomness each run


def load_catalog() -> list[dict]:
    try:
        products_df = pd.read_csv(CATALOG_FILE)
    except FileNotFoundError:
        print(f"Error: {CATALOG_FILE} not found. Run scraper.py and clean_data.py first.")
        sys.exit(1)

    if products_df.empty:
        print(f"Error: {CATALOG_FILE} is empty. Nothing to simulate transactions against.")
        sys.exit(1)

    if "Source" in products_df.columns:
        synthetic_rows = (products_df["Source"] != "scraped").sum()
        if synthetic_rows > 0:
            print(
                f"[Warning] {synthetic_rows} of {len(products_df)} catalog rows are marked "
                f"as non-scraped (synthetic/fallback) data."
            )
            print(
                "Generating transactions against non-real catalog data means "
                "raw_sales_data.csv will reference fabricated products. "
                "Re-run the scraper for real data, or proceed only if this is "
                "intentional (e.g. for a demo/test run)."
            )
            answer = input("Proceed anyway? [y/N]: ").strip().lower()
            if answer != "y":
                print("Aborting.")
                sys.exit(1)
    else:
        print(
            f"[Notice] {CATALOG_FILE} has no 'Source' column, so catalog provenance "
            f"can't be verified. Proceeding, but consider adding that column upstream."
        )

    return products_df.to_dict("records")


def build_customers(rng: np.random.Generator, fake: Faker) -> list[dict]:
    """Draw each customer's acquisition date and latent behavioral traits."""
    lam_scale = (LAMBDA_MEAN_PER_YEAR / 365.0) / LAMBDA_SHAPE
    hazard_scale = 1.0 / (LIFETIME_MEAN_DAYS * (DROPOUT_SHAPE - 1))

    customers = []
    for i in range(NUM_CUSTOMERS):
        # Acquisition spread across the window so cohorts are real cohorts.
        acquired_offset = int(rng.integers(0, WINDOW_DAYS))
        # Pareto/NBD dropout: hazard first, then an exponential lifetime.
        hazard = float(rng.gamma(DROPOUT_SHAPE, hazard_scale))
        lifetime = float(rng.exponential(1.0 / hazard)) if hazard > 0 else float(WINDOW_DAYS)
        customers.append({
            "CustomerID": f"CUST_{1000 + i}",
            "CustomerName": fake.name(),
            "Email": fake.email(),
            "acquired": START_DATE + timedelta(days=acquired_offset),
            # purchases/day while active
            "lambda": float(rng.gamma(LAMBDA_SHAPE, lam_scale)),
            # dropout hazard per day, and the lifetime it implies
            "dropout_hazard": hazard,
            "lifetime_days": lifetime,
        })
    return customers


def generate_transactions(products: list[dict], customers: list[dict],
                          rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = []
    used_invoice_ids = set()

    def new_invoice() -> str:
        while True:
            candidate = f"INV_{rng.integers(100000, 1000000)}"
            if candidate not in used_invoice_ids:
                used_invoice_ids.add(candidate)
                return candidate

    for customer in customers:
        # The acquisition purchase is what makes them a customer.
        purchase_days = [0.0]

        # Subsequent purchases: Poisson process, truncated at dropout.
        churn_day = customer["lifetime_days"]
        max_day = (END_DATE - customer["acquired"]).days
        t = 0.0
        rate = customer["lambda"]
        if rate > 0:
            while True:
                t += float(rng.exponential(1.0 / rate))
                if t > churn_day or t > max_day:
                    break
                purchase_days.append(t)

        for offset in purchase_days:
            product = products[int(rng.integers(0, len(products)))]
            invoice_date = customer["acquired"] + timedelta(days=int(offset))

            quantity = int(rng.choice(QUANTITY_CHOICES, p=QUANTITY_WEIGHTS))
            base_price = float(product["Price"])
            multiplier = 1 + float(rng.uniform(-PRICE_VARIANCE_PCT, PRICE_VARIANCE_PCT))

            transactions.append({
                "InvoiceNo": new_invoice(),
                "CustomerID": customer["CustomerID"],
                "CustomerName": customer["CustomerName"],
                "ProductID": product["Product_ID"],
                "ProductName": product["Product_Name"],
                "UnitPrice": round(base_price * multiplier, 2),
                "Quantity": quantity,
                "InvoiceDate": invoice_date.strftime("%Y-%m-%d"),
            })

    sales_df = pd.DataFrame(transactions).sort_values("InvoiceDate").reset_index(drop=True)

    counts = sales_df.groupby("CustomerID").size()
    truth_df = pd.DataFrame([{
        "CustomerID": c["CustomerID"],
        "acquired": c["acquired"].strftime("%Y-%m-%d"),
        "true_lambda_per_year": round(c["lambda"] * 365, 4),
        "true_dropout_hazard_per_day": round(c["dropout_hazard"], 6),
        "true_lifetime_days": round(c["lifetime_days"], 1),
        "churned_before_window_end": bool(
            c["acquired"] + timedelta(days=c["lifetime_days"]) < END_DATE),
        "observed_purchases": int(counts.get(c["CustomerID"], 0)),
    } for c in customers])

    return sales_df, truth_df


def summarize(sales_df: pd.DataFrame, truth_df: pd.DataFrame) -> None:
    per_customer = sales_df.groupby("CustomerID")["InvoiceNo"].nunique()
    revenue = (sales_df["UnitPrice"] * sales_df["Quantity"]).sum()
    print(f"  transactions      {len(sales_df):,}")
    print(f"  active customers  {sales_df['CustomerID'].nunique():,} "
          f"of {len(truth_df):,} generated")
    print(f"  orders/customer   {per_customer.mean():.2f} avg, "
          f"{per_customer.max()} max")
    print(f"  repeat rate       {(per_customer >= 2).mean()*100:.1f}% "
          f"(placed 2+ orders)")
    print(f"  churned in window {truth_df['churned_before_window_end'].mean()*100:.1f}%")
    print(f"  total revenue     ${revenue:,.2f}")


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    fake = Faker()
    Faker.seed(RANDOM_SEED)

    products = load_catalog()
    customers = build_customers(rng, fake)
    sales_df, truth_df = generate_transactions(products, customers, rng)

    sales_df.to_csv(OUTPUT_FILE, index=False)
    truth_df.to_csv(TRUTH_FILE, index=False)

    print(f"Generated {OUTPUT_FILE}:")
    summarize(sales_df, truth_df)
    print(f"Wrote {TRUTH_FILE} (latent traits, for model validation).")
    print("Customer identities and transaction history are fully synthetic by design.")


if __name__ == "__main__":
    main()
