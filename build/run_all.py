"""End-to-end build: dimensions -> facts -> raw mirror -> parquet -> docs."""
from __future__ import annotations

import sys

from . import config as C
from . import (dictionary, dimensions, facts, lifecycle_doc, postprocess, raw_mirror,
               refresh_docs, validation, vendor_map)
from .utils import ensure_dirs, log, run_step, write_parquet

SOURCE_MAPPING = """# Source-to-target change log

How each element of the Philippine BNPL source (`tyme_data_download/tyme`) was
re-expressed for a US credit-card issuer.

## 1. Business model

| Source (BNPL) | Target (credit card) |
|---|---|
| BNPL account with a spending limit | Revolving credit-card account with credit limit, APR and statement cycle |
| 4-instalment "DYNAMIC_TERM" product, 0.74% fee | Six card products (Secured, Student, Signature, Platinum, Gold, Business) with APRs 20.99%-26.99% |
| Voucher / QR-code checkout | Card authorisation at POS or e-commerce, with chip / contactless / swipe / ECOM entry modes |
| `bnpl_loan_account` instalment ledger | `fact_installment_plan` + `fact_installment_schedule` (optional pay-over-time), plus `fact_statement` for the revolving line |
| `bnpl_payment_workflow` bank-payment state machine | `fact_payment` plus `card_payment_workflow` raw mirror |
| No interest, no statements | Monthly statement with opening/closing balance, minimum due, interest, fees and delinquency |
| No delinquency concept | DPD and delinquency buckets (CURRENT to 120+), charge-off at 180 DPD |

## 2. Currency and amount calibration

| Source | Target |
|---|---|
| PHP, all 4,241 purchases | USD |
| PHP median ticket 2,458.38 | **USD median ticket ≈ $85**, calibrated to US card norms |
| PHP p95 6,699.15 | USD p95 ≈ $600 |
| No FX fields | `currency_code`, `original_amount`, `fx_rate` for cross-border spend |

Amounts are **not** carried over 1:1. That rule was evaluated first: at 15
transactions per account per month (the target frequency) a 1:1 median ticket of
$2,458 implies about **$37,000 of monthly spend per card**, which exceeds a
plausible consumer credit limit and caused ~50% of authorisations to be declined.
The ticket distribution is therefore re-expressed in US norms, anchored on a
median of $85 and a p95 of $600, while preserving MCC ordering (travel and
electronics above fast food and digital).

Transaction frequency is a deliberate departure from the source: the BNPL book runs
2.82 purchases per account per month; the credit-card book targets a median of
**15 per month**, segmented by customer behaviour (see §2.1).

### 2.1 Behavioural segmentation

Customers are scored on a propensity blend (income 45%, credit score 30%, product
tier 15%, age 10%), ranked, and cut into fixed shares. The weighted median lands on
15 transactions per month.

| Segment | Share | Median txns/month | Profile |
|---|---:|---:|---|
| AFFLUENT_POWER | 10% | 40 | high-income premium cardholder |
| PRIME_ACTIVE | 25% | 20 | prime, multi-category spender |
| CORE_MAINSTREAM | 35% | 15 | mainstream prime/near-prime |
| LIGHT_USER | 22% | 6 | low usage, often revolving |
| DORMANT | 8% | 1.5 | near-inactive |

Within each segment a lognormal shock (sigma 0.30-0.45) produces outliers, so a
heavy user can reach 80-100 transactions in a month while a dormant account can
have none.

## 3. Geography

| Source | Target |
|---|---|
| Country `PH` (100% of rows) | Country `US` for domestic rows; 8 international cities for cross-border |
| 61 Philippine cities (Quezon City, Cebu, Makati, Bonifacio Global City...) | Top-25 US metros, 69 cities, 236 ZIP-level locations |
| `BANK_BIC` PAPHPHM1XXX, PAEYPHM2XXX, MBTCPHMMXXX, BNORPHMMXXX, GXCHPHM2XXX | US banks: JPMorgan Chase, Bank of America, Wells Fargo, Citi, US Bank, PNC, Capital One, TD, Ally, Schwab |
| `COUNTRY_OF_BIRTH` PH/TH/SA | US, with a small share of foreign-born customers retained via `city_of_birth` |
| MCC mix dominated by 5411 (grocery), 5262, 6012, 5311, 5912 | 47 MCCs weighted to a US card portfolio, adding 5812 dining, 5541 fuel, 4121 rideshare, 4511 airlines, 5815/5816 digital |
| Merchant names: Lazada, Robinsons, SM Supermarket, Mercury Drug, Shell | 20,000 merchant locations across 86 fictional US brands and 47 MCCs (FreshLine Market, Urban Table, Northstar Fuel, Skyway Airlines, CloudStream...) |

## 4. Customer identity

| Source | Target |
|---|---|
| Hashed PH full name / e-mail / TIN | US first/last name, e-mail, phone, masked SSN (non-issuable 900-999 area range) |
| `PHILSYS_ID`, `UMID`, `DRIVER_LICENSE`, `PASSPORT`, `PROFESSIONAL_ID`, `POSTAL_ID`, `SSS_ID`, `VOTER_ID` | `DRIVER_LICENSE`, `STATE_ID`, `PASSPORT`, `ITIN`, `MILITARY_ID` |
| Income in PHP (median ₱30,000) | Income in USD at 1:1 (median $30,000), shaped by occupation prior |
| Gender 0/1 with 49% null | F/M with 0% null |
| Age from DOB, median 30.8 | Age 19-84, median ≈ 38 |
| No credit score | FICO-style score 300-850 with tier-consistent product assignment |
| No address | Street, city, state, ZIP, metro, region, lat/long, timezone |

## 5. Table-level renames

| Source table | Target star schema | Target raw mirror |
|---|---|---|
| `bnpl_onboarding` | `dim_customer` | `card_onboarding` |
| `bnpl_whitelist` | `dim_account` (limits) | `card_whitelist` |
| `bnpl_account` | `dim_account` | `card_account` |
| `bnpl_activation` | `dim_card` (activation) | `card_activation` |
| `bnpl_purchase` | `fact_transaction` | `card_purchase` |
| `bnpl_voucher_qr_code` | `dim_merchant`, `dim_location`, `fact_transaction` | `card_voucher_qr_code` |
| `bnpl_voucher` | `fact_transaction` | `card_voucher` |
| `bnpl_voucher_activity` | `fact_transaction` audit trail | `card_voucher_activity` |
| `bnpl_payment_option` | `fact_installment_plan` | `card_payment_option` |
| `bnpl_purchase_schedule` | `fact_installment_schedule` | `card_purchase_schedule` |
| `bnpl_purchase_activity` | `fact_installment_schedule` events | `card_purchase_activity` |
| `bnpl_loan_account` | `fact_installment_schedule` ledger | `card_loan_account` |
| `bnpl_payment_workflow` | `fact_payment` | `card_payment_workflow` |
| `bnpl_fraud_insights` | `fact_fraud_alert` | `card_fraud_insights` |
| `bnpl_fraud_upload` | `fact_fraud_alert` | `card_fraud_upload` |
| `profile_mapping` | `dim_customer` surrogate | `profile_mapping` |
| `segment_logs_android/ios/backend` | `fact_app_event` | `segment_logs_*` |
| *(new)* | `dim_date`, `dim_product`, `dim_channel`, `dim_location`, `dim_merchant`, `dim_card` | - |
| *(new)* | `fact_statement`, `fact_credit_limit_event`, `fact_delinquency_snapshot` | - |

## 6. Deliberate additions

The source had no statement, interest, delinquency, credit-limit or card-issuer
concepts. These were added because a credit-card book is not analytically
meaningful without them:

- monthly statement cycle with opening/closing balance, minimum due and due date
- interest accrual on revolving balances at the product APR
- annual fee, foreign transaction fee and late fee
- credit limit increases/decreases
- DPD and delinquency buckets, with charge-off at 180 DPD
- card-level lifecycle (physical/virtual, replacement)
- decline reasons and declined authorisations
- cross-border transactions with FX conversion
"""



def _sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_tree_sha256() -> str:
    import hashlib

    paths = sorted(
        [*(C.ROOT / "build").glob("*.py"), C.ROOT / "pyproject.toml"],
        key=lambda p: p.relative_to(C.ROOT).as_posix(),
    )
    h = hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(C.ROOT).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _write_manifest(failed: int, started_at: str) -> None:
    """Release manifest: hashes, counts, timestamps and the gate result."""
    import hashlib, json, platform, sys
    from datetime import datetime, timezone
    from importlib.metadata import version

    files = []
    for sub in ("dim", "fact", "raw"):
        for p in sorted((C.OUT / sub).glob("*.parquet")):
            files.append({"table": p.stem, "layer": sub, "bytes": p.stat().st_size,
                          "sha256": _sha256(p)})
    docs = []
    for p in sorted(C.DOC_DIR.glob("*")):
        if p.is_file():
            docs.append({"name": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)})

    finished_at = datetime.now(timezone.utc).isoformat()
    dependency_versions = {
        name: version(name) for name in ("numpy", "pandas", "pyarrow")
    }
    dependency_versions_sha256 = hashlib.sha256(
        json.dumps(dependency_versions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "dataset": "synthetic-us-credit-card",
        "output": str(C.OUT),
        "build_command": f"CC_OUT_DIR={C.OUT.name} python -m build.run_all",
        "started_at": started_at,
        "finished_at": finished_at,
        "generated_at": finished_at,
        "seed": C.SEED,
        "period": [str(C.PERIOD_START), str(C.PERIOD_END)],
        "scale": {"customers": C.N_CUSTOMERS, "merchants": C.N_MERCHANTS},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "generator_source_tree_sha256": _source_tree_sha256(),
        "dependency_versions": dependency_versions,
        "dependency_versions_sha256": dependency_versions_sha256,
        "validation_failures": failed,
        "released": failed == 0,
        "files": files,
        "docs": docs,
    }
    (C.OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for marker in ("RELEASE_OK", "RELEASE_FAILED"):
        (C.OUT / marker).unlink(missing_ok=True)
    if failed:
        (C.OUT / "RELEASE_FAILED").write_text(
            f"{failed} validation failure(s); do not use this build.", encoding="utf-8")
    else:
        (C.OUT / "RELEASE_OK").write_text(
            "All validation checks passed.", encoding="utf-8")
    log(f"manifest written (released={failed == 0})")


def main() -> int:
    from datetime import datetime, timezone

    started_at = datetime.now(timezone.utc).isoformat()
    ensure_dirs()
    log(f"Output directory: {C.OUT}")
    log(f"Scale: {C.N_CUSTOMERS:,} customers, {C.N_MERCHANTS:,} merchants")
    log("Building dimensions")
    dims = run_step("dimensions", dimensions.build_all)

    log("Building facts (sequential statement-cycle simulation)")
    facts_out = run_step("facts", facts.build_all, dims)

    # keep the finalised account dimension
    dims["dim_account"] = facts_out.pop("dim_account")

    log("Building raw operational mirror")
    raw = run_step("raw_mirror", raw_mirror.build_all, dims, facts_out)

    log("Writing parquet")
    written = []
    for name, df in dims.items():
        p = write_parquet(df, C.DIM_DIR / f"{name}.parquet", name)
        written.append((name, len(df), p.stat().st_size))

    for name, df in facts_out.items():
        p = write_parquet(df, C.FACT_DIR / f"{name}.parquet", name)
        written.append((name, len(df), p.stat().st_size))

    for name, df in raw.items():
        if df is None or df.empty:
            log(f"  skip empty raw table {name}")
            continue
        p = write_parquet(df, C.RAW_DIR / f"{name}.parquet", name)
        written.append((name, len(df), p.stat().st_size))

    # ---- gated direct write: postprocess, then validate the FINAL files ----
    log("Applying postprocessing (transaction hygiene, card lifecycle, network)")
    postprocess.main()

    log("Final validation and documentation on postprocessed files")
    refresh_docs.main()
    failed = validation.LAST_FAIL_COUNT

    log("Building source vendor rename map")
    try:
        vendor_map.run()
    except Exception as e:
        log(f"  vendor rename map skipped: {e}")

    (C.DOC_DIR / "source_to_target_mapping.md").write_text(SOURCE_MAPPING, encoding="utf-8")
    (C.DOC_DIR / "delinquency_extension.md").write_text(lifecycle_doc.TEXT, encoding="utf-8")
    _write_manifest(failed, started_at)

    total_rows = sum(n for _, n, _ in written)
    total_mb = sum(s for _, _, s in written) / 1024 / 1024
    log(f"Wrote {len(written)} tables, {total_rows:,} rows, {total_mb:.1f} MB")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
