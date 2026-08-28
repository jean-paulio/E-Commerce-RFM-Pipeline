"""
clean_data.py

Cleans the raw product catalog produced by scraper.py:
 - drops rows where Product_Name is actually just a price
 - drops rows that are generic UI/navigation text, not real products
 - writes the result to a NEW file (never overwrites the raw catalog)

This script never fabricates product data. If cleaning leaves too few
rows to be useful, it stops and tells you, rather than silently
injecting invented products into your dataset.
"""

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "product_catalog.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "product_catalog_clean.csv")
MIN_ACCEPTABLE_ROWS = 15

# Substrings (not exact matches) that indicate UI chrome rather than a product name
GARBAGE_SUBSTRINGS = [
    "menu",
    "sale",
    "shop now",
    "previous slide",
    "next slide",
    "back to school deal",
    "wayfair featured product",
]


def is_garbage_name(name: str) -> bool:
    lowered = name.lower()
    return any(term in lowered for term in GARBAGE_SUBSTRINGS)


def clean_catalog() -> None:
    df = pd.read_csv(INPUT_FILE)
    print(f"Original catalog size: {len(df)} products")

    # Drop rows where Product_Name is actually just a price string
    mask_price_as_name = df["Product_Name"].str.contains(r"\$", na=False)
    dropped_price_names = mask_price_as_name.sum()
    df_clean = df[~mask_price_as_name]

    # Drop rows that look like UI/navigation junk rather than real products
    mask_garbage = df_clean["Product_Name"].apply(is_garbage_name)
    dropped_garbage = mask_garbage.sum()
    df_clean = df_clean[~mask_garbage]

    print(
        f"Dropped {dropped_price_names} row(s) where Product_Name was a price, "
        f"{dropped_garbage} row(s) of UI/navigation text."
    )
    print(f"Cleaned catalog size: {len(df_clean)} products")

    if len(df_clean) < MIN_ACCEPTABLE_ROWS:
        print(
            f"\n[Warning] Cleaned catalog has only {len(df_clean)} products "
            f"(minimum expected: {MIN_ACCEPTABLE_ROWS})."
        )
        print(
            "This usually means the scraper's selectors didn't match the page "
            "structure well, or the source page had little real product content."
        )
        print(
            "Refusing to auto-generate replacement products, since that would "
            "silently mix fabricated data into what should be real catalog data."
        )
        print(
            "Next steps: check wayfair_page.html and scraper.py's card/price "
            "selectors, or re-save a fuller copy of the page, then re-run."
        )
        sys.exit(1)

    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"{OUTPUT_FILE} has been written with {len(df_clean)} cleaned products.")
    print(f"(Original {INPUT_FILE} was left untouched.)")


if __name__ == "__main__":
    clean_catalog()
