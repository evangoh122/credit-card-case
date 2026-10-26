# Synthetic US Credit Card Dataset — v6

A synthetic US credit-card portfolio derived from the Tyme (Philippines) BNPL
extract in `tyme_data_download/tyme`. The BNPL business model, Philippine
geography, PHP currency and Philippine identity data have been re-expressed as a
US card issuer, expanded into a star schema with an operational raw mirror.

**Period:** 2025-09-01 → 2026-09-30 (13 statement months)
**Customers:** 50,000
**Transactions:** 12,730,758
**Currency:** USD (cross-border rows carry a local currency and FX rate)
**Geography:** top-25 US metros, 69 cities, 244 locations (incl. 8 international), 20,002 merchants

## Version history

| Version | Status |
|---|---|
| `credit_card_data_v6/` | **current, released** — exit 0, 143 PASS / 0 FAIL |
| `credit_card_data_v5/` | abandoned mid-build |
| `credit_card_data_v4/` | rejected (ledger defects) |
| `credit_card_data_v3/` | rejected |
| `credit_card_data_v2/`, `credit_card_data/` | earlier prototypes, 5k scale |

See `hand_off_v6.md` for the full release record and reviewer responses.

---

## 1. Folder layout

```
credit_card_data_v6/
|-- dim/                 8 dimension tables
|-- fact/               10 fact tables
|-- raw/                19 operational-mirror tables (sampled, connected)
|-- docs/
|   |-- validation_report.md
|   |-- data_dictionary.md / .csv
|   |-- source_to_target_mapping.md
|   |-- source_vendor_rename_map.csv
|   |-- delinquency_extension.md
|   `-- table_inventory.csv
|-- release_manifest.json     hashes, counts, gate status
`-- RELEASE_OK                absent if any validation failed
```

---

## 2. Schema

### Dimensions

| Table | Grain | Key | Rows |
|---|---|---|---:|
| `dim_customer` | customer | `customer_id` | 50,000 |
| `dim_account` | credit-card account | `account_id` | 54,425 |
| `dim_card` | physical/virtual card | `card_id` | 61,518 |
| `dim_merchant` | merchant location | `merchant_id` | 20,002 |
| `dim_location` | city/ZIP geography | `geo_id` | 244 |
| `dim_product` | card product | `product_id` | 6 |
| `dim_channel` | channel / entry mode | `channel_id` | 7 |
| `dim_date` | calendar day (2025-2029) | `date_key` | 1,826 |

### Facts

| Table | Grain | Rows |
|---|---|---:|
| `fact_transaction` | authorisation attempt | 12,730,758 |
| `fact_statement` | account × statement cycle | 584,963 |
| `fact_payment` | posted payment (card or instalment) | 1,027,375 |
| `fact_payment_allocation` | payment → statement allocation | 1,093,401 |
| `fact_installment_plan` | pay-over-time plan | 137,361 |
| `fact_installment_schedule` | plan instalment | 987,315 |
| `fact_installment_allocation` | payment → instalment allocation | 549,559 |
| `fact_credit_limit_event` | limit change | 2,580 |
| `fact_delinquency_snapshot` | account × statement date | 584,963 |
| `fact_fraud_alert` | fraud alert | 29,309 |
| `fact_app_event` | digital event | 4,131,808 |

### Relationships

```
dim_date
dim_customer --< dim_account --< dim_card
                    |
                    +--< fact_transaction -- dim_merchant -- dim_location
                    |        |
                    |        +-- fact_installment_plan --< fact_installment_schedule
                    |        |                              ^
                    |        |                              |
                    |        |          fact_installment_allocation
                    |        |                              |
                    |        +-- fact_fraud_alert            |
                    +--< fact_statement                        |
                    |        ^                                 |
                    |        |                                 |
                    +--< fact_payment_allocation >-- fact_payment
                    +--< fact_credit_limit_event
                    +--< fact_delinquency_snapshot
                    `--< fact_app_event
```

---

## 3. How it is generated

1. **Dimensions** — customers scored on income, credit score, product tier and age,
   then cut into behavioural segments (median 15 purchases/account-month overall).
2. **Per statement cycle** — a vectorised draw of merchant, MCC, amount, channel,
   hour and currency; monthly volume is overdispersed (lognormal multiplier plus
   spike months).
3. **Authorisation** — capped by available credit; random declines; fraud event
   decided *before* settlement (blocked attempts are declined and carry no loss).
4. **Statement** — opening + charges + interest + fees − payments = closing; single
   oldest-due-first payment waterfall; every allocation recorded.
5. **Instalments** — financed principal leaves the revolving ledger; repayments are
   real cash rows with an allocation bridge.
6. **Delinquency** — measured on the unpaid minimum; `CURRENT` / `PAST_DUE` (1-60) /
   `DELINQUENT` (61-180) / `CHARGE_OFF` (>180).

---

## 4. Rebuilding

```powershell
$env:CC_OUT_DIR="credit_card_data_v7"
python -m build.run_all       # postprocess + final validation run inside this command
```

The build exits non-zero if any validation check fails, and writes
`release_manifest.json` plus `RELEASE_OK` or `RELEASE_FAILED`.

Requirements: Python 3.14, pandas 2.3.3, numpy 2.5.4, pyarrow 25.0.1,
polars 2.0.0 (see `pyproject.toml`). Postprocessing and the ML extracts read the
470 MB `fact_transaction.parquet` with Polars (lazy scan / streaming sink) rather
than loading it whole into pandas, which OOM'd v7; `run_all` also frees the
in-memory tables after writing them. Regression: `tests/test_postprocess_polars.py`. `refresh_docs`/validation/dictionary also profile it lazily
(`validation.tx_metrics`, parquet metadata row counts); the churn and credit-default
ML extracts keep label-defining columns (`customer_status`, `churn_date`,
`account_status`, `closed_date`) out of the features, enforced by
`ml_extracts.check_no_label_columns`. Credit-default `status_at_obs` / `paid_full_share_3m` are recomputed from
payment allocations dated on or before the observation date (the final statement
`payment_status` is settled with later cash); `max_source_date` includes payment
dates. Tests: `tests/test_v7_asof_payments.py`. Tests: `tests/test_v7_polars_fixes.py`. Runtime ≈ 55 minutes for 50k customers.

---

## 5. Querying (DuckDB)

```sql
SELECT d.year_month,
       COUNT(*)             AS txns,
       SUM(t.amount_usd)    AS spend
FROM read_parquet('credit_card_data_v6/fact/fact_transaction.parquet') t
JOIN read_parquet('credit_card_data_v6/dim/dim_date.parquet') d USING (date_key)
WHERE t.status = 'APPROVED' AND t.transaction_type = 'PURCHASE'
GROUP BY 1 ORDER BY 1;

SELECT delinquency_status, COUNT(*) AS accounts, SUM(balance) AS balance
FROM read_parquet('credit_card_data_v6/fact/fact_delinquency_snapshot.parquet')
WHERE month_key = '2026-09'
GROUP BY 1 ORDER BY 1;

SELECT network, COUNT(*) AS txns, ROUND(SUM(amount_usd), 0) AS spend
FROM read_parquet('credit_card_data_v6/fact/fact_transaction.parquet')
WHERE status = 'APPROVED'
GROUP BY 1 ORDER BY spend DESC;
```

---

## 6. Caveats

- **Amounts are calibrated to US norms** (median $85, p95 $600); a 1:1 PHP→USD
  carry-over was rejected because at 15 transactions/month it implies ~$37k of
  monthly spend per card.
- **Frequency is segmented and overdispersed** — median 14 per account-month,
  p95 66, max 450.
- **No causal structure.** Limits affect approved spend mechanically; λ is
  untouched; there is no treatment assignment. See `hand_off_v6.md` §12.
- **Fraud is separable by construction** — loss is calibrated to 6.94 bps of spend,
  but the label derives from a score, so a model will separate it easily.
- **`RETURNED` payments are not exercised.**
- **DPD is observed at statement dates** (interval-censored).
- **No collections/recovery cash flows.**
- **Raw layer is a sample** (~1.2% of transactions), internally connected.
- **Identifiers are synthetic but not reserved-range** (ordinary email domains and
  ordinary-looking US phone numbers).
