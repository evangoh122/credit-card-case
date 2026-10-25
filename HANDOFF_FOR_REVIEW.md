> **SUPERSEDED - historical record only.**
> This document describes an earlier candidate and is retained for audit.
> The current release record is `feedback_v6.md` and `hand_off_v6.md`.

# Handoff for independent review — v2

**Project:** Synthetic US credit-card portfolio derived from a Philippine BNPL extract
**Build:** `credit_card_data_v4/` (supersedes `v3`, which had material defects)
**Period:** 2025-09-01 → 2026-09-30 (13 statement months)
**Status:** build complete, exits 0, **92 PASS / 16 INFO / 0 FAIL**
**This document:** what was built, what changed after the first review, and what remains open.

---

## 0. Read this first — what changed after review

The first review rejected `v3`. It was right. The 88/88 PASS in the previous handoff was misleading because the checks only proved statements balanced *internally*; they never reconciled statements back to transactions or payments.

| # | Finding | Status | Fix |
|---|---|---|---|
| 1 | Zero-activity statements reused prior cycle values | **Fixed** | `cyc_*` reset every cycle |
| 1b | `lifetime_purchases` double-counted (2.02× inflated) | **Fixed** | single increment |
| 2 | Two incompatible payment allocations | **Fixed** | one waterfall + new `fact_payment_allocation` bridge |
| 3 | Final-statement payments dropped | **Fixed** | pending payments flushed after the last cycle |
| 4 | Instalments never advanced (all ACTIVE) | **Fixed** | evaluated as of period end; COMPLETED / DELINQUENT / ACTIVE |
| 5 | Card validity ignored (230k pre-issue transactions) | **Fixed** | per-transaction card by issue date; `replaced_date` |
| 5b | Churn closed only the primary account | **Fixed** | churn closes every account |
| 5c | Charge-offs had no closed date | **Fixed** | `closed_date` set; 5 charge-offs in v4 |
| 6 | Raw mirror lost sign/type | **Fixed** | signed amounts, `REFUND_CONFIRMED`, `transactionType` |
| 6b | Unseeded `uuid4` broke reproducibility | **Fixed** | deterministic `det_uuid(rng)` |
| 6c | Validation never aborted the build | **Fixed** | exit code 1 on any FAIL |
| 6d | Row-count checks vacuously passed | **Fixed** | reclassified as `INFO` |
| 6e | No dependencies declared | **Fixed** | `pyproject.toml` core/research/MLOps/synthesis extras |
| 7 | Stale docs (1:1 text, 727 merchants, corrupted values) | **Fixed** | README, REVIEW_GUIDE, vendor summary |
| — | "$1 tolerance" rounding mismatch (found during re-test) | **Fixed** | exact-cent comparison in simulator and validator |

Two findings were **validator** defects rather than data defects, both now corrected:
a \$1 materiality threshold used by the simulator but 1 cent in the validator (70 snapshots),
and a `minimum_due > 1` filter that excluded sub-dollar minimums the simulator correctly counted (4 snapshots).

---

## 1. Executive summary

| Entity | Count |
|---|---:|
| Customers | 50,000 |
| Accounts | 54,501 |
| Cards | 61,995 |
| Merchants | 20,002 |
| Locations | 236 |
| **Transactions** | **12,696,823** |
| Statements | 585,328 |
| Payments | 478,976 |
| **Payment allocations** | **1,094,598** |
| Pay-over-time plans | 138,262 |
| Plan instalments | 995,403 |
| Credit-limit events | 2,233 |
| Delinquency snapshots | 585,328 |
| Fraud alerts | 127,829 |
| App events | 4,130,310 |
| **Total rows** | **27,304,639** |
| **Parquet size** | **1,794 MB** |

Selected facts: median ticket \$85.04 (target 85), p95 \$602.62 (target 600),
median 14 transactions per account-month, approval rate 85.8%, 30+ DPD 1.59%,
fraud alerts 1.17% of approvals, net fraud loss \$14.70M, 5 charge-offs.

---

## 2. What was requested vs delivered

| Requirement | Delivered | Where |
|---|---|---|
| Synthetic data from the Tyme BNPL extract | Yes | `build/`, `credit_card_data_v4/` |
| BNPL → credit card | Yes | `docs/source_to_target_mapping.md` |
| Change all customer data | Yes | `dim_customer` |
| Change all location data | Yes | `dim_location`, `dim_merchant` |
| Currency → USD | Calibrated to US norms | §6 |
| Expand to a longer window | 13 months | config |
| Fact + dimension tables | 8 dims, 10 facts | §4 |
| All Parquet | Yes, snappy | `dim/`, `fact/`, `raw/` |
| Scale | 50,000 customers, 12.7M transactions | this build |
| Vendor names changed, categories kept | 385 source vendors mapped; 20,002 US merchants | §9 |
| Visa / Mastercard / Amex | Yes, plus Appendix D | §10 |
| Random monthly volume | Overdispersed, max 450 | §8 |
| Document default/cure | Spec written | `docs/delinquency_extension.md` |
| Causal + statistics research | Planned on a v5 DGP | §17 |

---

## 3. Source audit

| Table | Rows | Note |
|---|---:|---|
| `bnpl_purchase` | **4,241** | the only true transaction table |
| `bnpl_purchase_activity` | 23,775 | instalment status events |
| `bnpl_purchase_schedule` | 16,177 | instalment schedule |
| `bnpl_loan_account` | 446,537 | instalment ledger |
| `bnpl_payment_workflow` | 17,157 | payment state machine |
| `bnpl_voucher` | 11,751 | QR voucher lifecycle |
| `bnpl_voucher_qr_code` | 1,475 | merchant/location payload |
| `bnpl_onboarding` | 957 | 616 profiles |
| `bnpl_whitelist` | 1,700 | 1,081 limit offers |
| `bnpl_account` | 6,936 | 435 accounts |
| `segment_logs_*` | 374,101 | app events |

Source statistics: 4,241 purchases, 289 transacting accounts, 2025-01-30 → 2025-07-05,
2.82 purchases per account-month, median 2,458.38 PHP, p25 1,811.00, p75 3,759.21,
p95 6,699.15, max 24,030.00, **385 vendors inside `METADATA` JSON, 41 MCCs**,
`MERCHANT_ID` constant `"2"`, `STORE_ID` 100% null.

**Known source quirk:** `bnpl_purchase_*.parquet` also matches
`bnpl_purchase_activity_*` and `bnpl_purchase_schedule_*`, which once inflated
4,241 purchases to 44,193 and corrupted an early amount anchor. All globs now match
numeric suffixes only.

---

## 4. Target model

### 4.1 Dimensions

| Table | Grain | Key | Rows |
|---|---|---|---:|
| `dim_customer` | customer | `customer_id` | 50,000 |
| `dim_account` | credit-card account | `account_id` | 54,501 |
| `dim_card` | card | `card_id` | 61,995 |
| `dim_merchant` | merchant location | `merchant_id` | 20,002 |
| `dim_location` | city/ZIP | `geo_id` | 236 |
| `dim_product` | product | `product_id` | 6 |
| `dim_channel` | channel/entry mode | `channel_id` | 7 |
| `dim_date` | calendar day | `date_key` | 730 |

### 4.2 Facts

| Table | Grain | Rows |
|---|---|---:|
| `fact_transaction` | authorisation attempt | 12,696,823 |
| `fact_statement` | account × statement cycle | 585,328 |
| `fact_payment` | posted payment | 478,976 |
| `fact_payment_allocation` | payment → statement allocation | 1,094,598 |
| `fact_installment_plan` | pay-over-time plan | 138,262 |
| `fact_installment_schedule` | plan instalment | 995,403 |
| `fact_credit_limit_event` | limit change | 2,233 |
| `fact_delinquency_snapshot` | account × statement date | 585,328 |
| `fact_fraud_alert` | fraud alert | 127,829 |
| `fact_app_event` | digital event | 4,130,310 |

`fact_payment_allocation` is the audit trail for the waterfall: one row per
(payment, statement, amount). It is what makes `paid_amount`, `payment_status`,
`is_late` and DPD provably consistent.

### 4.3 Relationships

```
dim_date ─────────────┐
dim_customer ──< dim_account ──< dim_card
      │                │
      │                ├──< fact_transaction ──> dim_merchant ──> dim_location
      │                │         │
      │                │         ├──> fact_installment_plan ──< fact_installment_schedule
      │                │         └──> fact_fraud_alert
      │                ├──< fact_statement ──< fact_payment_allocation >── fact_payment
      │                ├──< fact_credit_limit_event
      │                └──< fact_delinquency_snapshot
      └──< fact_app_event
```

### 4.4 Sign conventions

`amount_usd` signed (purchases positive, refunds negative); `status` is
APPROVED/DECLINED/REVERSED; declined rows never move a balance.

---

## 5. Pipeline and code map

| File | Responsibility |
|---|---|
| `build/config.py` | catalogues, shares, MCC profiles, 25 metros, segments, calibration |
| `build/utils.py` | per-step RNG streams, hashing, dates, Parquet IO |
| `build/dimensions.py` | date, location, product, channel, merchant, customer, account, card |
| `build/txgen.py` | vectorised per-cycle transaction draw |
| `build/facts.py` | `simulate_v2` — statement/payment/waterfall simulator |
| `build/raw_mirror.py` | `card_*` / `segment_logs_*` operational mirror |
| `build/vendor_map.py` | 385 source vendors → US brands, MCC preserved |
| `build/validation.py` | 108 checks (92 PASS + 16 INFO) |
| `build/dictionary.py` | data dictionary |
| `build/postprocess.py` | auth-code hygiene, recurring flags, card cancellation, `network` |
| `build/refresh_docs.py` | re-validate from Parquet without re-simulating |
| `build/lifecycle_doc.py` | default/cure specification |
| `build/run_all.py` | orchestration; **exits 1 on any FAIL** |

Determinism: master seed `20260924`, independent `SeedSequence` per step.
All raw identifiers are seeded; `uuid.uuid4()` is no longer used.

Runtime: dimensions 48 s, facts 24 min, raw mirror 34 min, validation + docs 8 min.

---

## 6. Currency decision history

| Attempt | Rule | Median ticket | Outcome |
|---|---|---|---|
| 1 | FX at 58.14 PHP/USD | \$12 | implausibly small |
| 2 | 1:1 PHP→USD | \$2,458 | at 15 txns/month = \$37k monthly spend; ~50% declines |
| **3 (adopted)** | US-normal calibration | **\$85** | \$1,275/month, 85.8% approval |

Attempt 2 was requested and implemented, then rejected on arithmetic. The ticket
distribution is fitted so the MCC-weighted mixture has median \$85 and p95 \$600,
preserving category ordering.

---

## 7. Segmentation and frequency

| Segment | Share | Config median | Realised median |
|---|---:|---:|---:|
| AFFLUENT_POWER | 10% | 40 | 49.0 |
| PRIME_ACTIVE | 25% | 20 | 24.0 |
| CORE_MAINSTREAM | 35% | 15 | 18.2 |
| LIGHT_USER | 22% | 6 | 7.3 |
| DORMANT | 8% | 1.5 | 1.8 |

Monthly counts: `Poisson(λ·m)`, `m ~ lognormal(0, 0.60)`, plus 1.5% spike months ×2.5–6.
Realised distribution: p5 2, p25 7, p50 14, p75 27, p95 66, p99 127, max 450.
20.3% of account-months ≤ 5 transactions.

---

## 8. Merchant model

20,002 locations, 86 fictional brands, 47 MCCs, ~635 transactions per merchant,
3.5% international. Names unique (`Brand #00001`). The 385 source vendors are mapped
to US brands with **MCC and category unchanged** in
`docs/source_vendor_rename_map.csv`; the generated book does not reuse them.

---

## 9. Card networks

| Network | Accounts | Credit limit | Outstanding | Approved txns | Spend |
|---|---:|---:|---:|---:|---:|
| Visa | 26,831 | \$371.7M | \$92.12M | 5,258,968 | \$834.0M |
| Mastercard | 22,066 | \$229.5M | \$70.15M | 4,024,695 | \$630.0M |
| American Express | 5,604 | \$127.8M | \$28.62M | 1,608,074 | \$257.5M |

---

## 10. Balances, payments and delinquency

### 10.1 Payment waterfall

Every payment is allocated **oldest due first**: minima, then billed balances. Every
allocation is written to `fact_payment_allocation`, and statement `paid_amount`,
`payment_status`, `is_late` and the delinquency snapshot all derive from that single
mechanism. Payments scheduled by the final statement are flushed after the loop, so
the final statement is not left artificially pending.

### 10.2 Statement payment behaviour

| Status | Share |
|---|---:|
| `PAID_FULL` | 81.0% |
| `PAID_PARTIAL` (minimum) | 6.9% |
| `PENDING` (due after window) | 6.8% |
| `NO_BALANCE` | 4.2% |
| `UNPAID` | 1.0% |
| `PARTIAL` (below minimum) | 0.2% |

### 10.3 Delinquency

| Bucket | Share |
|---|---:|
| CURRENT | ~90.1% |
| 1-29 | 8.4% |
| 30-59 | 1.3% |
| 60-89 | 0.2% |
| 90+ | ~0.0% |

30+ DPD = **1.59%**. Charge-offs: **5** (180 DPD), all with a `closed_date`.

### 10.4 Reconciliation

| Check | Result |
|---|---|
| Statement arithmetic | max delta **\$0.00** |
| Statements vs underlying purchases | max delta **\$0.00** |
| Statements vs underlying refunds | max delta **\$0.00** |
| `paid_amount` vs allocated payments | max delta **\$0.00** |
| Zero-activity statements carrying activity | **0** |
| Over-allocated statements | **0** |
| DPD reconstructed from waterfall | **0 days mismatch** |
| Current balance = last statement − later payments | **100%** within \$1 |

---

## 11. Raw operational mirror

19 tables, source idioms preserved (DynamoDB wrappers, nested JSON, hashed IDs,
`METADATA`). Amounts are **signed**, refunds carry `REFUND_CONFIRMED`, and all
identifiers are seeded.

The transaction-level raw tables are **sampled** at this scale (150k transactions,
30k plans, 60k payments, 20k fraud rows) so the deliverable stays portable; the star
schema is complete and unsampled.

---

## 12. Validation

**92 PASS, 16 INFO, 0 FAIL.** Full report: `docs/validation_report.md`.

New checks added after review (none of which existed in v3):

- statements reconcile to underlying purchases and refunds
- no zero-activity statement carries activity
- `paid_amount` equals allocated payments
- no statement over-allocated beyond amount due
- every payment targets an existing statement
- delinquency snapshot reconstructed from the waterfall
- no transaction before card issue, after replacement, after closure, after churn
- charge-offs carry a closed date
- instalments: paid rows have dates; no past-due PENDING
- completed plans have zero remaining balance; no repaid plan left ACTIVE
- allocation FKs resolve

Row counts are `INFO`, not PASS.

---

## 13. Known limitations

1. **No causal structure.** See §17.
2. **Fraud is too separable.** Drawn from a heuristic score; a model will reach
   AUC ≈ 0.98. Not credible as an ML portfolio dataset without hardening.
3. **Only 5 charge-offs**, so the charge-off path is exercised but thin.
4. **No prior-year history**; `lifetime_*` covers 13 months.
5. **Pre-period opening balances are drawn**, not simulated.
6. **Pay-over-time plans do not post back to the revolving balance.**
7. **Raw mirror is sampled** at transaction level.
8. **App events are loosely coupled** to transactions.
9. **No collections/recovery cash flows.**
10. **Timezone handling is naive**; the merchant carries the IANA zone.
11. **Privacy claim corrected:** identifiers are synthetic but use ordinary domains
    (gmail/yahoo/outlook) and ordinary-looking US phone numbers, **not** reserved
    `example.com` or 555 ranges. Do not present them as reserved-range data.

---

## 14. Delinquency lifecycle extension

`docs/delinquency_extension.md`: cure/roll matrices, default at 180 DPD, recovery
curves, proposed `fact_delinquency_transition`, config placeholders
(`DELINQUENCY_LIFECYCLE_ENABLED=False`, `DEFAULT_DPD=180`, `CURE_WINDOW_DAYS=30`).

---

## 15. MLOps roadmap

ML decides **who the customer is**; the deterministic simulator decides **what they
do**. Faker for identities, DeepEcho for account-level population synthesis (never
transaction sequences), YData only for fraud-class augmentation inside `ml/`.
Dataset hardening required first: fraud rate to 0.15–0.25%, label noise, a
fraud-type regime shift, and out-of-time + out-of-entity splits.

---

## 16. Causal inference and statistics research agenda

### 16.1 Corrected framing (this was wrong in the previous handoff)

The earlier claim that "limit increase → spend is exactly zero" was **incorrect**.

- λ (the spending-intensity parameter) does **not** depend on the limit, so the
  *direct intensity effect* is zero by construction.
- But limits determine the **authorisation ceiling**, so a limit increase raises
  **approved** spend for constrained accounts. That is a real mechanical effect,
  observable in `fact_transaction`, and it must not be mistaken for behavioural
  elasticity.

Causal work therefore needs to separate (a) the mechanical approval-margin effect,
(b) any behavioural response, and (c) selection into limit increases.

### 16.2 Required v5 DGP changes

| Element | Design |
|---|---|
| Treatment | `limit_increase_offer`, assigned with confounding by utilisation, tenure, segment |
| Effect | known heterogeneous effect on spend **and** explicit approval-margin channel |
| Second experiment | randomised cashback promo 50/50 with known lift |
| Third experiment | pay-over-time offer with propensity assignment and known delinquency effect |
| Instrument | score-threshold rule, **but** identified as fuzzy RD with an explicit exclusion-restriction argument — a threshold alone is not a valid IV |
| Churn | new churn mechanism that can respond to fraud alerts (currently churn is assigned before fraud exists) |
| Ground truth | `docs/dgp_manifest.json` |

### 16.3 Studies

| # | Question | Identification |
|---|---|---|
| S1 | Limit increase → approved spend | event study, TWFE, Callaway–Sant'Anna, Double ML |
| S2 | Pay-over-time → delinquency | PSM, IPWRA, E-values |
| S3 | Fraud alert → churn | Cox PH, Fine–Gray (requires the v5 churn mechanism) |
| S4 | Promo → spend | diff-in-means, CUPED, power analysis |
| S5 | Estimator recovery | 500-draw Monte Carlo: bias, RMSE, CI coverage |

S5 demonstrates **internal validity under the assumed DGP**, not realism or external
validity. The write-up must say so.

### 16.4 Statistics track

EVT/tail index for amounts; Kaplan–Meier and Cox with censoring; roll rates as a
Markov chain; STL seasonality; bootstrap/jackknife CIs for portfolio KPIs; power
analysis; Benjamini–Hochberg FDR; Bayesian hierarchical delinquency; synthetic-data
validity (TSTR, nearest-neighbour, membership inference).

---

## 17. Reproduction

```bash
$env:CC_OUT_DIR="credit_card_data_v5"
python -m build.run_all          # exits 1 on any validation failure
python -m build.postprocess
python -m build.refresh_docs
```

Python 3.14, pandas 2.3.3, numpy 2.5.4, pyarrow 25.0.1. Seed `20260924`.

**Note:** `python -m build.run_all | Tee-Object` masks the exit code in PowerShell;
read `$LASTEXITCODE` from the python process or run without the pipe.

---

## 18. File inventory

```
credit_card_data_v4/
├── dim/          8 tables     9.7 MB
├── fact/        10 tables   746.3 MB
├── raw/         19 tables 1,037.7 MB
└── docs/
    ├── data_dictionary.csv / .md
    ├── validation_report.md
    ├── source_to_target_mapping.md
    ├── source_vendor_rename_map.csv (386 vendors)
    ├── source_vendor_rename_summary.md
    ├── delinquency_extension.md
    └── table_inventory.csv
```

---

## 19. Review checklist

| # | Question | Where |
|---|---|---|
| 1 | Do statements reconcile to transactions **and** payments? | validation §reconciliation |
| 2 | Is DPD consistent with the waterfall? | validation §delinquency |
| 3 | Are card and account lifecycles respected? | validation §temporal |
| 4 | Is the instalment state as-of period end? | validation §installments |
| 5 | Is the 85.8% approval defensible? | §6, Appendix D |
| 6 | Is the US-normal calibration right vs 1:1? | §6 |
| 7 | Is the raw mirror faithful at 1.2% sampling? | §11 |
| 8 | Does any Philippine artefact survive? | validation checks |
| 9 | Is fraud too separable for ML? | §13.2, §15 |
| 10 | Is the v5 causal design identifiable? | §16 |
| 11 | Are Monte Carlo plans statistically sound? | §16.3 |
| 12 | Any PII/compliance overclaim? | §13.11 |

---

*All figures from `credit_card_data_v4`, generated 2026-10-25.*
