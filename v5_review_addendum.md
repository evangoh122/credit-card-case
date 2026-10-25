> **SUPERSEDED - historical record only.**
> This document describes an earlier candidate and is retained for audit.
> The current release record is `feedback_v6.md` and `hand_off_v6.md`.

# V5 review addendum

*Appended to `build_v5_log.txt` after the build completes. Kept as a standalone
file (`v5_review_addendum.md`) so reviewers can cite it directly.*

---

## 1. Provenance

| Item | Value |
|---|---|
| Output directory | `credit_card_data_v5/` |
| Generator | `build/` at the commit/state described below |
| Seed | `20260924` |
| Period | 2025-09-01 → 2026-09-30 (13 statement months) |
| Scale | 50,000 customers, 20,002 merchants |
| Reviewer 1 (Kimi) | blocking findings on ledger consistency, fail gate, payment semantics, lifecycle |
| Reviewer 2 (Codex) | additional gaps on date coverage, raw referential integrity, geography, closure semantics, version skew |

**This build supersedes v3 and v4. Neither should be used for analysis.**
v4 remains on disk only as the rejected candidate.

---

## 2. What changed between v4 and v5

### 2.1 From Kimi's review (blocking)

| # | Finding | v5 status |
|---|---|---|
| 1 | Zero-activity statements reused prior cycle values | fixed — cycle aggregates reset |
| 1b | `lifetime_purchases` double-counted | fixed — ratio to actual is 1.000 |
| 2 | Two incompatible payment allocations | fixed — single waterfall + `fact_payment_allocation` |
| 3 | Final-statement payments dropped | fixed — flushed after the loop |
| 4 | Instalments never advanced | fixed — evaluated as of period end |
| 5 | Card validity ignored | fixed — per-transaction card by issue/replacement/expiry |
| 5b | Churn closed only the primary account | fixed |
| 5c | Charge-offs had no closed date | fixed |
| 6 | Raw mirror lost sign/type | fixed — signed amounts, `REFUND_CONFIRMED` |
| 6b | Unseeded `uuid4` | fixed — deterministic `det_uuid` |
| 6c | Validation could not fail the build | fixed — `Report.fail_count` drives exit code 1 |
| 6d | Row-count checks vacuously passed | fixed — reclassified as `INFO` |

### 2.2 From Codex's review

| # | Finding (verified against v4) | v4 evidence | v5 fix |
|---|---|---|---|
| 1 | `dim_date` ended before instalments | 222,472 rows (22.3%) outside | dimension extended to 2029-12-31 (1,826 rows) |
| 2 | Raw schedule/purchase sampled independently | 213,441 of 215,880 orphan references | connected sampling: plans kept only if their purchase is in the sampled extract |
| 3 | International merchants had null geography | 149 merchants, 111,456 transactions | 8 international geo members added; every merchant/txn resolves |
| 4 | USD non-international at international merchants | 108,674 rows | `is_international` now derives from merchant geography |
| 5 | Credit balances hidden as `NO_BALANCE` | 14,628 statements / −$45.58M; 1,890 accounts / −$6.41M | new `CREDIT_BALANCE` status; `dim_account.credit_balance` |
| 6 | Post-closure payments and card issuance | 1,250 payments / $4.65M; 101 cards | flush capped at `closed_date`; cards skipped after closure |
| 7 | Version skew between code, parquet and docs | v4 parquet lacked `unapplied_amount` | v5 rebuilt and all docs regenerated from this output |
| 8 | Blank `max delta` detail | PowerShell `$` interpolation | restored in validator |

### 2.3 Payment semantics (Kimi finding 2, extended)

The waterfall now allocates **oldest due first**, capped at each statement's
**incremental billed amount** (which includes pre-period opening debt on the first
statement). This removes the double-counting that arises from carried balances.

- `fact_payment_allocation` records every allocation.
- `fact_payment.unapplied_amount` records credit not applied to any statement.
- `fact_statement.paid_amount` = allocations to that statement.
- `fact_statement.paid_closing_amount` = portion of the closing balance settled by
  later payments (FIFO view). **`PAID_FULL` is defined on this field.**
- `is_late` = the minimum was not covered by a payment dated on or before the due date.

Enforced identities:

```
sum(allocations per payment) + unapplied_amount = payment.amount_usd
sum(allocations per statement)                = statement.paid_amount
PAID_FULL  =>  paid_closing_amount >= closing_balance
paid_amount <= billed_amount
```

### 2.4 Credit policy staging (requested by the client)

| DPD | `delinquency_status` |
|---|---|
| 0 | `CURRENT` |
| 1–60 | `PAST_DUE` |
| 61–180 | `DELINQUENT` |
| >180 | `CHARGE_OFF` |

Charge-off is strictly **>180 DPD** (previously `>=180`). The stage is published on
`fact_delinquency_snapshot` and `dim_account`, and the validator checks the mapping.

---

## 3. Validation gate

- `Report.check()` increments `fail_count` on every `FAIL`.
- `validation.run()` publishes `LAST_FAIL_COUNT`.
- `build.run_all` returns exit code **1** if any check fails.
- Row counts are `INFO`, not `PASS`.

The gate is proven by construction: during v5 development, a run with one real
failure reported `FAIL: 1` and `LAST_FAIL_COUNT 1`.

---

## 4. Independent reproduction commands

Run these against `credit_card_data_v5/` — they do not use the validator:

```python
import pandas as pd, numpy as np
B = "credit_card_data_v5/"

tx = pd.read_parquet(B + "fact/fact_transaction.parquet")
st = pd.read_parquet(B + "fact/fact_statement.parquet")
al = pd.read_parquet(B + "fact/fact_payment_allocation.parquet")
pay = pd.read_parquet(B + "fact/fact_payment.parquet")

# 1. statements reconcile to transactions
pur = tx[(tx.transaction_type == "PURCHASE") & (tx.status == "APPROVED")]
agg = pur.groupby("statement_id").amount_usd.sum()
c = st.set_index("statement_id").purchases_amount
print("purchase delta", (c - agg.reindex(c.index).fillna(0)).abs().max())

# 2. paid_amount equals allocations
a = al.groupby("statement_id").amount_usd.sum()
c2 = st.set_index("statement_id").paid_amount
print("allocation delta", (c2 - a.reindex(c2.index).fillna(0)).abs().max())

# 3. cash identity
ps = al.groupby("payment_id").amount_usd.sum()
c3 = pay.set_index("payment_id")[["amount_usd", "unapplied_amount"]]
print("cash delta", (c3.amount_usd - ps.reindex(c3.index).fillna(0) - c3.unapplied_amount).abs().max())

# 4. PAID_FULL settled
pf = st[st.payment_status == "PAID_FULL"]
print("PAID_FULL short", (pf.paid_closing_amount < pf.closing_balance - 0.01).sum())

# 5. date dimension coverage
dd = pd.read_parquet(B + "dim/dim_date.parquet", columns=["date_key"])
sch = pd.read_parquet(B + "fact/fact_installment_schedule.parquet", columns=["date_key"])
print("schedule outside dim_date", (~sch.date_key.isin(set(dd.date_key))).sum())

# 6. geography
m = pd.read_parquet(B + "dim/dim_merchant.parquet", columns=["geo_id"])
print("merchants without geo", m.geo_id.isna().sum())

# 7. raw referential integrity
rp = pd.read_parquet(B + "raw/card_purchase.parquet", columns=["ID"])
rs = pd.read_parquet(B + "raw/card_purchase_schedule.parquet", columns=["PURCHASE_ID"])
print("raw orphan schedules", (~rs.PURCHASE_ID.isin(set(rp.ID))).sum())
```

Expected: all deltas `0.0`, all counts `0`.

---

## 5. What v5 does **not** claim

1. **No causal structure.** Limits affect approved spend mechanically through the
   authorisation ceiling, but λ is untouched; there is no treatment assignment.
2. **Fraud is too separable** for a credible ML benchmark (AUC ≈ 0.98) until the
   dataset-hardening work in the roadmap.
3. **Raw layer is a sample**, now internally connected, but not a full mirror.
4. **No collections/recovery cash flows.**
5. **Only a handful of charge-offs** (strictly >180 DPD), so that path is thin.
6. **No prior-year history.**
7. **App events are loosely coupled** to transactions.
8. **Privacy wording**: identifiers are synthetic but use ordinary domains and
   ordinary-looking US phone numbers — not reserved ranges.

---

## 6. For the next reviewer

Please check, in this order:

1. the three cash identities in §4 reproduce at `0.0`
2. `PAID_FULL` never exceeds `closing_balance`
3. every `fact_installment_schedule.date_key` joins `dim_date`
4. every merchant and transaction has a `geo_id`
5. raw schedules resolve to raw purchases
6. no post-closure payments or card issuance
7. `delinquency_status` matches the 60/180 policy
8. the build log ends with a validation summary and `EXITCODE=0`

If any of those fail, the build should be rejected again.

---

*End of addendum. Generated as part of the v5 release; append to `build_v5_log.txt`
once the build process releases the file.*
