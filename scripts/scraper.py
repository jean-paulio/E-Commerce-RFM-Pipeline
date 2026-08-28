"""
scraper.py

Parses a locally saved Wayfair HTML page and extracts product names and
prices into product_catalog.csv.

NOTE: This script only reads a local HTML snapshot (wayfair_page.html).
It does not make live HTTP requests. If live fetching is added later,
be sure to respect Wayfair's robots.txt / terms of service and add
appropriate rate limiting.

TITLE/PRICE SELECTION STRATEGY
Primary: select by CSS class. Inspecting the real page showed every product
title lives in an <h2> with the exact class set TITLE_CLASSES, and every
current price lives in a <span> with the exact class set PRICE_CLASSES.
This is far more reliable than guessing from text content.

Caveat: these class names (e.g. "_6o3atzbl") look like hashed/atomic CSS
classes from a CSS-in-JS build system. They are very likely to change
whenever Wayfair ships a new frontend build, so this selector can go stale
without warning. If TITLE_MATCHES/PRICE_MATCHES both come back empty, that's
the signal to re-run the class discovery (see find_title_and_price_classes.py
if you build one) rather than assume the page has no products.

Fallback: if the class-based pass finds nothing, fall back to the old
text-content heuristic (badge-word filtering + longest-candidate selection)
so the script still produces something usable if Wayfair's markup changes.
"""

import hashlib
import os
import re
from typing import Optional

from bs4 import BeautifulSoup
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SOURCE_FILE = os.path.join(DATA_DIR, "wayfair_page.html")
OUTPUT_FILE = os.path.join(DATA_DIR, "product_catalog.csv")
FALLBACK_FILE = os.path.join(DATA_DIR, "product_catalog_SAMPLE.csv")  # clearly distinct from real output
MIN_NAME_LENGTH = 10  # short strings are almost always badges ("Sale", "Menu"), not titles
MAX_NAME_LENGTH = 80

# Verified against a real saved Wayfair page (see module docstring). These
# are exact-match class SETS -- a tag must have ALL of these classes, not
# just one, to count as a title/price element.
TITLE_CLASSES = {"_6o3atzbl", "_6o3atz18y", "_1lxwj2q1"}
PRICE_CLASSES = {"_6o3atzbl", "_6o3atzc7", "_6o3atz1d9", "_6o3atz1bd"}

# Matches things like "$45.00", "$1,299", "$45.00 - $65.00" (captures the first price only)
PRICE_PATTERN = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")

# Text that shows up in badge/UI spans, not real product titles. Only used
# by the text-heuristic fallback path.
BADGE_TERMS = [
    "menu",
    "sale",
    "shop now",
    "previous slide",
    "next slide",
    "back to school deal",
    "wayfair featured product",
    "deal",
    "clearance",
    "% off",
    "free shipping",
    "free delivery",
    "day delivery",
    "account dropdown",
    "open account",
    "sign in",
    "sign up",
    "add to cart",
    "add to registry",
]
BADGE_PREFIXES = ["by "]


def has_exact_classes(tag, required: set) -> bool:
    return required.issubset(set(tag.get("class") or []))


def looks_like_badge(text: str) -> bool:
    lowered = text.lower().strip()
    if any(term in lowered for term in BADGE_TERMS):
        return True
    if any(lowered.startswith(prefix) for prefix in BADGE_PREFIXES):
        return True
    return False


def truncate_name(name: str, max_length: int) -> str:
    """Trim to max_length on a word boundary rather than cutting mid-word."""
    if len(name) <= max_length:
        return name
    truncated = name[:max_length].rsplit(" ", 1)[0]
    return truncated.rstrip(" -,")


def find_title_fallback(card) -> Optional[str]:
    """Text-heuristic fallback: longest h2/h3/span candidate that isn't a price or badge."""
    candidates = []
    for el in card.find_all(["h2", "h3", "span"]):
        text = el.get_text(strip=True)
        if not text or "$" in text or looks_like_badge(text):
            continue
        candidates.append(text)
    return max(candidates, key=len) if candidates else None


def stable_product_id(name: str, price: float) -> str:
    """Generate a deterministic ID so the same product gets the same ID across runs."""
    digest = hashlib.md5(f"{name}|{price}".encode("utf-8")).hexdigest()[:8]
    return f"PROD_{digest}"


def extract_price(text: str) -> Optional[float]:
    """Extract the first valid dollar amount from a string. Returns None if not found."""
    match = PRICE_PATTERN.search(text)
    if not match:
        return None
    numeric_str = match.group(1).replace(",", "")
    try:
        value = float(numeric_str)
        return value if value > 0 else None
    except ValueError:
        return None


def parse_by_class(soup: BeautifulSoup) -> list[dict]:
    """Primary strategy: select titles and prices by their known CSS classes."""
    titles = [h2 for h2 in soup.find_all("h2") if has_exact_classes(h2, TITLE_CLASSES)]

    seen = set()
    products = []
    for title_el in titles:
        name = title_el.get_text(strip=True)
        if not name or len(name) < MIN_NAME_LENGTH:
            continue

        price = None
        for ancestor in title_el.parents:
            price_span = next(
                (s for s in ancestor.find_all("span") if has_exact_classes(s, PRICE_CLASSES)),
                None,
            )
            if price_span:
                price = extract_price(price_span.get_text(strip=True))
                break
            if ancestor.name == "body":
                break

        if price is None:
            continue

        name = truncate_name(name, MAX_NAME_LENGTH)
        key = (name, price)
        if key in seen:
            continue
        seen.add(key)

        products.append({
            "Product_ID": stable_product_id(name, price),
            "Product_Name": name,
            "Price": price,
            "Source": "scraped",
        })

    return products


def parse_by_heuristic(soup: BeautifulSoup) -> list[dict]:
    """Fallback strategy: guess title/price from any div/article card by text content."""
    cards = soup.find_all(["div", "article"])
    seen = set()
    products = []

    for card in cards:
        price_node = card.find(string=lambda t: t and "$" in t)
        if not price_node:
            continue
        price = extract_price(price_node.strip())
        if price is None:
            continue

        name = find_title_fallback(card)
        if not name or len(name) < MIN_NAME_LENGTH:
            continue

        name = truncate_name(name, MAX_NAME_LENGTH)
        key = (name, price)
        if key in seen:
            continue
        seen.add(key)

        products.append({
            "Product_ID": stable_product_id(name, price),
            "Product_Name": name,
            "Price": price,
            "Source": "scraped",
        })

    return products


def parse_local_wayfair() -> None:
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Could not find {SOURCE_FILE}. Please save the webpage to the data/ folder first.")
        return

    print("Reading local HTML file...")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    print("Attempting class-based extraction (primary strategy)...")
    product_list = parse_by_class(soup)

    if product_list:
        print(f"Class-based extraction found {len(product_list)} products.")
    else:
        print(
            "Class-based extraction found nothing -- Wayfair's markup may have "
            "changed. Falling back to text-heuristic extraction."
        )
        product_list = parse_by_heuristic(soup)
        print(f"Heuristic fallback found {len(product_list)} products.")

    df = pd.DataFrame(product_list).drop_duplicates(subset=["Product_Name", "Price"])

    if not df.empty:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully compiled {OUTPUT_FILE} with {len(df)} products.")
    else:
        # IMPORTANT: fallback/sample data is written to a DIFFERENT file and
        # explicitly labeled, so it can never be mistaken for real scraped data.
        print("No products could be parsed from the page structure.")
        print(f"Writing a clearly-labeled sample dataset to {FALLBACK_FILE} instead.")
        fallback_data = [
            {
                "Product_ID": stable_product_id(f"Sample Area Rug {i}", 0),
                "Product_Name": f"Sample Area Rug {i}",
                "Price": round(45.0 + i * 7.35, 2),
                "Source": "synthetic_fallback",
            }
            for i in range(1, 31)
        ]
        pd.DataFrame(fallback_data).to_csv(FALLBACK_FILE, index=False)
        print(
            f"Generated {FALLBACK_FILE} with 30 synthetic placeholder items. "
            f"{OUTPUT_FILE} was NOT modified."
        )


if __name__ == "__main__":
    parse_local_wayfair()

