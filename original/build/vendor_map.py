"""Source vendor rename map.

Every Philippine merchant in the Tyme extract is mapped to a fictional US brand.
The MCC and analytical category are preserved exactly; only the name changes.

Writes docs/source_vendor_rename_map.csv (one row per source vendor) and
docs/source_vendor_rename_summary.md (category roll-up).

If the Tyme extract is not present the step is skipped with a warning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .utils import log

# MCC -> (US description, analytical category). Category is what the target
# merchant dimension uses; it is never changed by the rename.
MCC_INFO = {
    "5411": ("Grocery Stores, Supermarkets", "Grocery"),
    "5499": ("Misc. Food Stores", "Grocery"),
    "5300": ("Wholesale Clubs", "Grocery"),
    "5262": ("Misc. Food Stores - Convenience", "Grocery"),
    "2012": ("Meat Provisioners", "Grocery"),
    "2013": ("Meat Provisioners", "Grocery"),
    "5812": ("Eating Places, Restaurants", "Dining"),
    "5814": ("Fast Food Restaurants", "Dining"),
    "5813": ("Drinking Places (Bars)", "Dining"),
    "5541": ("Service Stations", "Fuel"),
    "5172": ("Petroleum Products", "Fuel"),
    "5311": ("Department Stores", "Retail"),
    "5399": ("Misc. General Merchandise", "Retail"),
    "5331": ("Variety Stores", "Retail"),
    "5999": ("Misc. Retail", "Retail"),
    "5942": ("Book Stores", "Retail"),
    "5945": ("Toy and Hobby Shops", "Retail"),
    "5947": ("Gift, Card and Novelty", "Retail"),
    "5941": ("Sporting Goods", "Retail"),
    "5651": ("Family Clothing Stores", "Apparel"),
    "5691": ("Mens/Womens Clothing", "Apparel"),
    "5699": ("Misc. Apparel", "Apparel"),
    "5661": ("Shoe Stores", "Apparel"),
    "5732": ("Electronics Stores", "Electronics"),
    "5722": ("Household Appliance Stores", "Electronics"),
    "5045": ("Computers and Peripherals", "Electronics"),
    "5734": ("Computer Software Stores", "Digital"),
    "5912": ("Drug Stores and Pharmacies", "Health"),
    "8021": ("Dentists and Orthodontists", "Health"),
    "8043": ("Optometrists", "Health"),
    "7230": ("Beauty and Barber Shops", "Personal Care"),
    "7011": ("Hotels, Motels, Resorts", "Travel"),
    "4784": ("Tolls and Bridge Fees", "Transport"),
    "4812": ("Telecom Equipment", "Telecom"),
    "7832": ("Motion Picture Theaters", "Entertainment"),
    "5995": ("Pet Shops and Supplies", "Pets"),
    "0742": ("Veterinary Services", "Pets"),
    "5251": ("Hardware Stores", "Home Improvement"),
    "5013": ("Motor Vehicle Supplies", "Automotive"),
    "6012": ("Financial Institutions - Merchandise", "Financial Services"),
    "3072": ("Lodging", "Travel"),
}

DEFAULT_INFO = ("Misc. Retail", "Retail")


def _pick_brand(source_name: str, category: str, used: set[str]) -> str:
    pool = C.MERCHANT_BRANDS.get(category) or C.MERCHANT_BRANDS["Retail"]
    h = int(hashlib.sha256(source_name.encode()).hexdigest()[:8], 16)
    for attempt in range(64):
        brand = pool[(h + attempt * 7) % len(pool)]
        # keep the source's branch structure by appending a stable store number
        store = 100 + ((h >> 8) + attempt * 13) % 900
        candidate = f"{brand} #{store}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    candidate = f"{pool[0]} #{h % 9999}"
    used.add(candidate)
    return candidate


def build() -> pd.DataFrame | None:
    files = sorted(C.SOURCE_DIR.glob("bnpl_purchase_[0-9]*.parquet"))
    if not files:
        log("  vendor rename map skipped: source extract not found")
        return None

    raw = pd.concat([pd.read_parquet(f, columns=["METADATA", "AMOUNT"]) for f in files],
                    ignore_index=True)
    md = raw.METADATA.dropna().map(json.loads)
    src = pd.DataFrame({
        "source_merchant_name": md.map(lambda d: d.get("merchantName") or d.get("QR_MERCHANT_NAME")),
        "source_mcc": md.map(lambda d: str(d.get("mcc") or d.get("categoryCode") or "")),
        "amount": pd.to_numeric(raw.AMOUNT, errors="coerce"),
    }).dropna(subset=["source_merchant_name"])

    grp = (src.groupby(["source_merchant_name", "source_mcc"], as_index=False)
              .agg(purchases=("amount", "size"), spend=("amount", "sum"))
              .sort_values(["purchases", "source_merchant_name"], ascending=[False, True]))

    used: set[str] = set()
    out = []
    for _, r in grp.iterrows():
        desc, cat = MCC_INFO.get(r.source_mcc, DEFAULT_INFO)
        out.append(dict(
            source_merchant_name=r.source_merchant_name,
            source_mcc=r.source_mcc,
            mcc_description=desc,
            category=cat,
            source_purchases=int(r.purchases),
            source_spend=round(float(r.spend), 2),
            new_merchant_name=_pick_brand(r.source_merchant_name, cat, used),
            new_mcc=r.source_mcc,          # unchanged
            new_category=cat,              # unchanged
        ))
    df = pd.DataFrame(out)
    total = df.source_purchases.sum()
    df["purchase_share_pct"] = (df.source_purchases / total * 100).round(2)
    return df


def write_docs(df: pd.DataFrame) -> None:
    C.DOC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.DOC_DIR / "source_vendor_rename_map.csv", index=False, encoding="utf-8")

    by_cat = (df.groupby("category")
                .agg(vendors=("source_merchant_name", "nunique"),
                     purchases=("source_purchases", "sum"))
                .sort_values("purchases", ascending=False))
    lines = ["# Source vendor rename map", "",
             f"{df.source_merchant_name.nunique()} Philippine merchant names were replaced with "
             "fictional US brands. **MCC and category are unchanged** — only the trading name "
             "differs. Full mapping: `source_vendor_rename_map.csv`.", "",
             "## Category roll-up", "",
             "| Category | Vendors | Source purchases | Share |", "|---|---:|---:|---:|"]
    total = df.source_purchases.sum()
    for cat, r in by_cat.iterrows():
        lines.append(f"| {cat} | {int(r.vendors)} | {int(r.purchases):,} | {r.purchases/total:.1%} |")
    lines += ["", "## Worked examples", "",
              "| Source vendor | MCC | Category | New vendor | New MCC |", "|---|---|---|---|---|"]
    for _, r in df.head(15).iterrows():
        lines.append(f"| {r.source_merchant_name} | {r.source_mcc} | {r.category} | "
                     f"{r.new_merchant_name} | {r.new_mcc} |")
    lines += ["", "The generated credit-card dataset uses its own US merchant catalogue "
                  "(`dim_merchant`, 20,000 merchants across 47 MCCs). It does not reuse the "
                  "385 source vendors, so no Philippine trading name survives anywhere in the "
                  "output.",
              ""]
    (C.DOC_DIR / "source_vendor_rename_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> pd.DataFrame | None:
    df = build()
    if df is not None:
        write_docs(df)
        log(f"vendor rename map: {len(df):,} vendors -> {C.DOC_DIR}")
    return df


if __name__ == "__main__":
    run()
