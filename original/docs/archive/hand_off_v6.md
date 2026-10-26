# Hand-off v6

**Project:** Synthetic US credit-card portfolio derived from a Philippine BNPL extract
**Build:** `credit_card_data_v6/`
**Status:** released - build exited 0, 143 PASS / 0 FAIL, released: true
**Author:** DeepSeek V4.1 Flash 1017 (provider deepseek, model ID deepseek-flash)
**Supersedes:** v3, v4 and v5 (all rejected)
**Reviews answered:** Kimi, Codex, MiMo

---

## 0. How to read this document

Section 1 states what changed. Sections 2–4 answer each reviewer point by point.
Section 5 lists the ledger identities the data now enforces. Section 6 describes the
schema changes. Section 11 is the **measured** result block and is the only section
that must be trusted over any claim elsewhere in the document.

If section 11 is empty or the release manifest says "released": false, **do not use the
build**.

---

## 1. What changed in v6

| Area | Change |
|---|---|
| Cycle accounting | `cyc_*` aggregates reset every cycle; `lifetime_purchases` double count removed |
| Payments | single oldest-due-first waterfall, capped at incremental billed debt; `fact_payment_allocation` records every allocation |
| Settlement | derived from **cash received after the cut**, not allocation sums — one payment cannot settle two statements |
| Final payments | flushed after the last statement; final DPD recomputed afterwards |
| Instalments | financed principal leaves the revolving ledger; repayments become real cash in `fact_payment` with `fact_installment_allocation` |
| International | transaction flag derives from merchant country; domestic merchants sampled only from US locations |
| Geography | 8 international location members; every merchant and transaction resolves to a `geo_id` |
| Churn | a churned customer always has a churn date; every account closes on or before it |
| Cards | per-transaction validity by issue, replacement and expiry; no card issued after closure |
| Fraud | event decided before settlement; blocked attempts are declined and carry no loss; loss calibrated to bps of spend |
| Refunds | `original_transaction_id` lineage |
| Limits | increases **and** decreases, applied from the next cycle so statement/snapshot limits agree |
| Delinquency | explicit 60/180 staging (`CURRENT` / `PAST_DUE` / `DELINQUENT` / `CHARGE_OFF`) |
| Raw layer | connected sampling for plans, schedules, vouchers and voucher activity; real card last-4 |
| Release | postprocess + final validation + docs inside one fail-closed run; `release_manifest.json` with hashes |

---

## 2. Response to MiMo

| # | Finding | Status | Detail |
|---|---|---|---|
| 1 | Fraud loss 10–20× industry | **fixed** | Probability retuned; validator asserts **3–12 bps** of approved spend. Alert-count band 0.1–0.8% |
| 2 | `auth_code` float64, 12× reuse | **fixed** | Zero-padded six-character strings; length validated |
| 3 | 114k dangling `dispute_id` | **fixed** | Column removed. Disputes are **not modelled**; stated in §12 |
| 4 | `CREDIT_BALANCE` unvalidated | **fixed** | Negative closing ⇔ `CREDIT_BALANCE`; `dim_account.credit_balance` identity validated |
| 5 | Duplicate emails/phones | **fixed** | Index-based email suffix; deterministic unique phone pattern |
| 6 | Validator blind spots | **fixed** | Minimum-due on negative closings; $0.01 tolerances; `applied_statement_id` = first allocation and validated |
| 7 | Documentation arithmetic | **fixed** | §11 separates star and raw totals; 8 dims / 10 facts; SSN range claim removed |

**One deliberate disagreement.** MiMo asks that over-allocation be measured against
`closing_balance`. With carried balances, closing double-counts prior statements'
debt, so the enforceable cap is the **incremental billed amount** (which includes
pre-period opening debt on the first statement). `paid_amount ≤ billed_amount` is
checked, and `paid_closing_amount` provides the closing-balance view. This is
documented in `build/validation.py` rather than silently changed.

---

## 3. Response to Codex

| # | Finding | Status | Detail |
|---|---|---|---|
| 1 | International classification not implemented | **fixed** | `is_intl = merchant.is_international`; transaction flag, merchant country and location country are cross-checked |
| 2 | Churn leaves active accounts | **fixed** | Customers without a feasible churn date stay active; all accounts of churned customers are closed on/before the date |
| 3 | Instalments are a parallel unreconciled ledger | **fixed** | Financed principal is excluded from revolving billing; repayments post to `fact_payment` (`payment_type='INSTALLMENT'`) with `fact_installment_allocation`; plan principal = financed purchases |
| 4 | Fraud decision/loss semantics inconsistent | **fixed** | Fraud event precedes settlement; `BLOCK` ⇒ declined and zero loss; loss only on settled confirmed fraud |
| 5 | Build gate does not certify the final data | **fixed** | `postprocess` and final validation run inside `run_all`; release manifest records hashes and gate status |
| — | Refund/reversal lineage | **fixed (refunds)** | `original_transaction_id`; reversals are not linked and are documented |
| — | Card expiry vacuous | **fixed** | Last card on an account is renewed only to period end; expiry is enforced per transaction |
| — | Credit-limit decreases absent | **fixed** | Risk-driven decreases implemented with event history |
| — | `utilization_pct` naming/cap | **documented** | Stored as a ratio, capped at 1.5; described in the data dictionary |
| — | Payment-state coverage thin | **open** | `RETURNED` is not exercised — see §12 |
| — | DPD sparse by construction | **documented** | Statement-date observation; interval censoring noted |
| — | Raw connectedness partial | **fixed** | Plans, schedules, vouchers, voucher activity and purchase activity all resolve |

---

## 4. Response to Kimi

| Finding | Status | Detail |
|---|---|---|
| Settlement double-counts allocations | **fixed** | Settlement is now `min(closing, cash received after the cut)`; independent of attribution |
| Final payments do not update account DPD | **fixed** | Final DPD recomputed after the flush |
| Geography repair creates contradictory domestic merchants | **fixed** | US-only sampling for domestic merchants; country/geo/flag cross-checks |
| Release gate validates pre-postprocess artefacts | **fixed** | Single orchestrated gate |
| First-cycle interest omitted | **fixed** | Interest accrues on opening debt from the first statement |
| Instalments ignore closure/charge-off | **fixed** | Rows due after closure are `CANCELLED`; plan status follows |
| Raw voucher/activity independently sampled | **fixed** | Activity is derived from the voucher sample (0 orphans) |
| Raw purchase-activity uses plan ID | **fixed** | Uses the originating transaction ID (0 orphans) |
| Refund lineage missing | **fixed** | `original_transaction_id` |
| Reviewer wording: `PAID_FULL` “short of”, not “exceed” | **fixed** | Checklist corrected in §13 |

---

## 5. Ledger identities now enforced

```
sum(allocations per payment) + payment.unapplied_amount = payment.amount_usd
sum(allocations per statement)                          = statement.paid_amount
statement.paid_amount                                  <= statement.billed_amount
statement.paid_closing_amount                          <= max(statement.closing_balance, 0)
PAID_FULL  =>  paid_closing_amount >= closing_balance
statement.opening_balance(i+1) = closing_balance(i) - payments_amount(i+1)
sum(approved non-financed purchases per statement)      = statement.purchases_amount
sum(refunds per statement)                              = statement.refunds_amount
sum(financed purchase amounts)                          = sum(plan.principal_amount)
sum(installment allocations per schedule)               = schedule.paid_amount
transaction.is_international                            = (merchant.country_code != "US")
transaction.merchant_geo_id                             = merchant.geo_id
```

Each identity is a validator check (see §10).

---

## 6. Schema changes

| Table | Change |
|---|---|
| `fact_transaction` | `is_installment_financed`, `original_transaction_id`, `fraud_flag`, `fraud_decision`, `fraud_case_status`, `network`, `auth_code` as string |
| `fact_payment` | `unapplied_amount`, `payment_type` (`CARD` / `INSTALLMENT`) |
| `fact_installment_allocation` | **new** — payment → schedule bridge |
| `fact_statement` | `billed_amount`, `paid_closing_amount`, `last_payment_date`, `payment_status` adds `CREDIT_BALANCE` and `CARRY_FORWARD` |
| `fact_delinquency_snapshot` | `delinquency_status` (`CURRENT` / `PAST_DUE` / `DELINQUENT` / `CHARGE_OFF`) |
| `dim_account` | `credit_balance`, `delinquency_status` |
| `dim_card` | `replaced_date` |
| `dim_location` | 8 international members (country_code ≠ US) |
| `dim_date` | extended to 2029-12-31 (1,826 rows) |
| `fact_fraud_alert` | `dispute_id` removed |

---

## 7. Credit policy (client-specified)

| DPD | Status | Meaning |
|---|---|---|
| 0 | `CURRENT` | up to date |
| 1–60 | `PAST_DUE` | past due but not delinquent |
| 61–180 | `DELINQUENT` | delinquent |
| >180 | `CHARGE_OFF` | charged off |

Charge-off is strictly **greater than** 180 DPD. The mapping is validated.

---

## 8. Product and portfolio configuration

| Item | Value |
|---|---|
| Customers | 50,000 |
| Accounts (≈1.09 per customer) | ~54,500 |
| Merchants | 20,002 across 47 MCCs, 86 brands |
| Period | 2025-09-01 → 2026-09-30 |
| Products | 6 (Secured, Student, Signature, Platinum, Gold, Business) |
| Networks | Visa ~50%, Mastercard ~40%, American Express ~10% |
| Segments | AFFLUENT_POWER 10%, PRIME_ACTIVE 25%, CORE_MAINSTREAM 35%, LIGHT_USER 22%, DORMANT 8% |
| Ticket | median $85, p95 $600 (US-normal calibration) |
| Frequency | median 14 per account-month, p95 66, max 450 |

---

## 9. Determinism and provenance

- Seed `20260924`; per-step `SeedSequence`; no unseeded UUIDs.
- `release_manifest.json` records: output path, timestamp, seed, period, scale,
  Python/platform, per-file sizes and SHA-256, documentation hashes, validation
  failure count and `released` boolean.
- `RELEASE_OK` / `RELEASE_FAILED` marker files make the gate unambiguous.
- Exact command sequence:

```powershell
$env:CC_OUT_DIR="credit_card_data_v6"
python -m build.run_all        # exits non-zero on any validation failure
```

Postprocessing and final validation run **inside** `run_all`; there is no separate
step that can drift.

---

## 10. Validation

The suite now covers 143 checks at test scale (plus 17 informational row counts).
Categories:

- primary keys (14 tables) and foreign keys (21 relationships)
- transaction ↔ statement reconciliation (purchases, refunds, conversions)
- payment cash identity, allocation completeness, applied-statement agreement
- settlement, `PAID_FULL`, over-allocation
- delinquency snapshot reconstruction and 60/180 staging
- instalment cash reconciliation and plan status
- merchant denormalisation, geography, international classification
- card issue/replacement/expiry, account closure, churn lifecycle
- credit-limit event direction and history
- fraud lifecycle (blocked ⇒ no loss; loss only when settled)
- refund lineage
- identity uniqueness
- date-dimension coverage for every fact

Row counts are `INFO`, not `PASS`. Any `FAIL` makes `run_all` return 1.

---

## 11. Measured results (v6)

Source: `credit_card_data_v6/release_manifest.json` and
`credit_card_data_v6/docs/validation_report.md`, generated by the same run.

### 11.1 Gate

| Metric | Value |
|---|---|
| Build exit code | **0** |
| Validation | **143 PASS, 17 INFO, 0 FAIL** |
| Manifest `released` | **true** |
| Marker file | `RELEASE_OK` present |
| Total tables / rows | 38 / **27,581,940** |
| Total size | **1,768.6 MB** |
| Star-schema rows | 22,047,420 |
| Seed / period | `20260924` / 2025-09-01 to 2026-09-30 |

### 11.2 Tables

| Table | Rows |
|---|---:|
| `fact_transaction` | 12,730,758 |
| `fact_app_event` | 4,131,808 |
| `fact_payment_allocation` | 1,093,401 |
| `fact_payment` | 1,027,375 |
| `fact_installment_schedule` | 987,315 |
| `fact_statement` | 584,963 |
| `fact_delinquency_snapshot` | 584,963 |
| `fact_installment_allocation` | 549,559 |
| `fact_installment_plan` | 137,361 |
| `fact_fraud_alert` | 29,309 |
| `fact_credit_limit_event` | 2,580 |
| `dim_card` | 61,518 |
| `dim_account` | 54,425 |
| `dim_customer` | 50,000 |
| `dim_merchant` | 20,002 |
| `dim_date` | 1,826 |
| `dim_location` | 244 |

### 11.3 Portfolio

| Metric | Value |
|---|---|
| Approved spend | **$1,712.5M** |
| Transactions by network | Visa 6,095,867; Mastercard 4,877,443; Amex 1,757,448 |
| Payment mix | PAID_FULL 84.0%; PAID_PARTIAL 3.6%; PENDING 6.7%; CREDIT_BALANCE 2.5%; NO_BALANCE 1.8%; UNPAID 1.2% |
| Payments by type | 477,816 card; 549,559 instalment |
| Financed purchases | $60.8M (excluded from the revolving ledger) |
| Negative-closing statements | 14,849 (all labelled `CREDIT_BALANCE`) |
| Total account credit | $5.67M |

### 11.4 Credit risk

| Metric | Value |
|---|---|
| Delinquency stage | CURRENT 89.17%; PAST_DUE 10.53%; DELINQUENT 0.29%; CHARGE_OFF 0.00% |
| 30+ DPD | 1.55% |
| 60+ DPD (delinquent) | 0.29% |
| Charge-offs | 4 |
| Closed accounts | 2,876 |

### 11.5 Fraud

| Metric | Value |
|---|---|
| Alerts | 29,309 |
| Blocked attempts | 11,748 |
| Net loss | $1,188,399 |
| **Loss rate** | **6.94 bps of approved spend** (industry 5-7 bps) |

## 12. Known limitations (not fixed, not hidden)

1. **No causal structure.** Limits affect approved spend mechanically through the
   authorisation ceiling; λ is untouched; no treatment assignment. Causal work needs
   a v7 DGP with known effects.
2. **Fraud is separable by construction.** Even at a credible loss rate, the label
   derives from a score; AUC will be high. Not suitable as an ML benchmark without
   the hardening described in the roadmap.
3. **`RETURNED` payments are not exercised.** NSF/returned-ACH behaviour and its
   delinquency effects are absent.
4. **DPD observed at statement dates only** — interval-censored; no daily hazard or
   cure-time survival modelling without explicit treatment.
5. **No collections or recovery cash flows** after charge-off.
6. **Reversals are not lineage-linked** (refunds are).
7. **Disputes/chargebacks are not modelled** (`dispute_id` removed).
8. **Raw layer is a sample** (~1.2% of transactions), now internally connected.
9. **Pay-over-time principals leave the revolving ledger entirely** — economically a
   point-of-sale conversion; there is no later balance-transfer event.
10. **No prior-year history**; `lifetime_*` covers 13 months.
11. **App events are loosely coupled** to transactions.
12. **Identifiers are synthetic but not reserved-range**: ordinary email domains and
    ordinary-looking US phone numbers. Do not present them as reserved-range data.

---

## 13. Reviewer checklist

| # | Check | Expected |
|---|---|---|
| 1 | `release_manifest.json` → `released` | `true` |
| 2 | Build exit code | `0` |
| 3 | Validation summary | 0 `FAIL` |
| 4 | `sum(alloc) + unapplied = payment` | 0.00 |
| 5 | `paid_amount = sum(allocations)` | 0.00 |
| 6 | `PAID_FULL ⇒ paid_closing_amount ≥ closing_balance` | 0 rows short |
| 7 | DPD reconstructed from cash | 0 days mismatch |
| 8 | Instalment cash = schedule paid | 0.00 |
| 9 | International flag = merchant country ≠ US | 0 mismatches |
| 10 | Every schedule `date_key` in `dim_date` | 0 missing |
| 11 | Every merchant/transaction has a `geo_id` | 0 nulls |
| 12 | Raw schedules/vouchers resolve | 0 orphans |
| 13 | No post-closure payment or card issuance | 0 rows |
| 14 | `delinquency_status` matches 60/180 | 0 mismatches |
| 15 | Net fraud loss | 3–12 bps of spend |

The wording point Kimi raised is corrected here: the critical test is that
`paid_closing_amount` is never **short of** a positive closing balance; the
upper-bound check is secondary.

---

## 14. Reproduction of the independent checks

```python
import pandas as pd
B = "credit_card_data_v6/"

tx = pd.read_parquet(B + "fact/fact_transaction.parquet")
st = pd.read_parquet(B + "fact/fact_statement.parquet")
al = pd.read_parquet(B + "fact/fact_payment_allocation.parquet")
ia = pd.read_parquet(B + "fact/fact_installment_allocation.parquet")
pay = pd.read_parquet(B + "fact/fact_payment.parquet")
sch = pd.read_parquet(B + "fact/fact_installment_schedule.parquet")

pur = tx[(tx.transaction_type == "PURCHASE") & (tx.status == "APPROVED")
         & (~tx.is_installment_financed)]
print("purchases", (st.set_index("statement_id").purchases_amount
      - pur.groupby("statement_id").amount_usd.sum().reindex(
          st.set_index("statement_id").index).fillna(0)).abs().max())

card = pay[pay.payment_type == "CARD"]
a = al.groupby("statement_id").amount_usd.sum()
print("allocations", (st.set_index("statement_id").paid_amount
      - a.reindex(st.set_index("statement_id").index).fillna(0)).abs().max())

ps = al.groupby("payment_id").amount_usd.sum()
c = card.set_index("payment_id")
print("cash", (c.amount_usd - ps.reindex(c.index).fillna(0) - c.unapplied_amount).abs().max())

isum = ia.groupby("schedule_id").amount_usd.sum()
print("instalment cash", (sch.set_index("schedule_id").paid_amount
      - isum.reindex(sch.set_index("schedule_id").index).fillna(0)).abs().max())

pf = st[st.payment_status == "PAID_FULL"]
print("paid full short", (pf.paid_closing_amount < pf.closing_balance - 0.01).sum())
```

---

*This hand-off is generated alongside `credit_card_data_v6`. If the build is
rejected, keep it as the record of what was attempted and what remains.*
