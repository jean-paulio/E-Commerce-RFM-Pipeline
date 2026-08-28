"""
simulator.py

Generates a simulated sales transaction history (raw_sales_data.csv) using
a REAL scraped product catalog and SIMULATED customers/transactions.

Customer and transaction data is synthetic by design, since real Wayfair
sales history is proprietary and not something we have (or should have)
access to. The product catalog, however, is expected to be real scraped
data -- this script checks for that and warns/aborts if the catalog looks
like fallback/sample data instead.
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

NUM_CUSTOMERS = 500
NUM_TRANSACTIONS = 2500
END_DATE = datetime(2026, 6, 1)
START_DATE = END_DATE - timedelta(days=365)
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


def generate_transactions(products: list[dict], fake: Faker) -> pd.DataFrame:
    customers = [
        {
            "CustomerID": f"CUST_{1000 + i}",
            "CustomerName": fake.name(),
            "Email": fake.email(),
        }
        for i in range(NUM_CUSTOMERS)
    ]

    transactions = []
    used_invoice_ids = set()

    for _ in range(NUM_TRANSACTIONS):
        customer = random.choice(customers)
        product = random.choice(products)

        random_days = random.randint(0, 365)
        invoice_date = START_DATE + timedelta(days=random_days)

        quantity = int(np.random.choice(QUANTITY_CHOICES, p=QUANTITY_WEIGHTS))

        # Small random variance so a year of transactions doesn't show one flat price
        base_price = float(product["Price"])
        price_multiplier = 1 + random.uniform(-PRICE_VARIANCE_PCT, PRICE_VARIANCE_PCT)
        unit_price = round(base_price * price_multiplier, 2)

        invoice_no = f"INV_{random.randint(100000, 999999)}"
        while invoice_no in used_invoice_ids:
            invoice_no = f"INV_{random.randint(100000, 999999)}"
        used_invoice_ids.add(invoice_no)

        transactions.append({
            "InvoiceNo": invoice_no,
            "CustomerID": customer["CustomerID"],
            "CustomerName": customer["CustomerName"],
            "ProductID": product["Product_ID"],
            "ProductName": product["Product_Name"],
            "UnitPrice": unit_price,
            "Quantity": quantity,
            "InvoiceDate": invoice_date.strftime("%Y-%m-%d"),
        })

    sales_df = pd.DataFrame(transactions)
    return sales_df.sort_values(by="InvoiceDate").reset_index(drop=True)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    fake = Faker()
    Faker.seed(RANDOM_SEED)

    products = load_catalog()
    sales_df = generate_transactions(products, fake)
    sales_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {OUTPUT_FILE} containing {len(sales_df)} simulated transactions.")
    print("Customer identities and transaction history are fully synthetic by design.")


if __name__ == "__main__":
    main()
