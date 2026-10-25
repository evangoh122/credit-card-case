# Feedback — dataset v6

## Reviewer provenance

| Field | Value |
|---|---|
| Dataset / release version | **v6** (`credit_card_data_v6/`) |
| Reviewer name | DeepSeek V4.1 Flash 1017 — automated release-validation reviewer |
| Model name | DeepSeek V4.1 Flash 1017 |
| Provider / model ID | `deepseek` / `deepseek-flash` |
| Review date | 2026-10-25 (build-host local time; all release timestamps in `release_manifest.json` are UTC) |
| Review type | Independent measurement of the completed release plus code inspection |
| Artifacts inspected | `credit_card_data_v6/` (38 Parquet tables), `release_manifest.json`, `docs/validation_report.md`, `docs/data_dictionary.md`, `build_v6_log.txt`, generator source in `build/` |
| Evidence basis | **Three separate tiers, never merged:** (a) independent Parquet measurements using a script that does **not** import `build/`; (b) static code inspection; (c) build log and release manifest. A built-in `PASS` is not accepted as verification of anything. |
| Modification policy | No generator, data or documentation was modified during this review. Documentation-only corrections made before this record are listed in §6. |

> This section is attributed to the model above. Any later reviewer must append
> their own separately attributed section; earlier sections are preserved verbatim.

---

## 1. Verdict

**Released. No release blockers found in the completed v6 artifact.**

| Gate | Result |
|---|---|
| Build exit code | **0** |
| Built-in validation | 143 PASS, 17 INFO, 0 FAIL |
| `release_manifest.json` → `released` | **true** |
| `RELEASE_FAILED` marker | absent |
| Independent measurement failures | **0** (24 checks, §3) |

The build is suitable for the analytical uses described in `hand_off_v6.md` §12,
subject to the accepted limitations in §5. It is **not** suitable for causal
inference, ML benchmarking without hardening, or installment-profitability work
that requires `RETURNED` payment behaviour.

---

## 2. Validity audit of prior reviewer feedback

The user asked specifically whether the earlier feedback was valid. Every material
claim was re-tested against the code and, where possible, against the released v6
data.

### 2.1 Kimi

| Finding | Validity | Evidence |
|---|---|---|
| Statement settlement double-counts allocations (`j <= i` sum) | **Valid** | Confirmed in the v5 code: allocations to older statements counted toward every newer statement's settlement. Replaced in v6 with `min(closing, cash received after the cut)`. |
| Final payments do not update account DPD | **Valid** | `last_dpd` was captured before the flush. v6 recomputes after every eligible payment; `dim_account.days_past_due` now matches. |
| Geography repair created contradictory domestic merchants | **Valid** | `us_geo` sampled the whole location dimension, including the 8 international rows. v6 samples only `country_code = 'US'`. |
| Release gate validates pre-postprocess artefacts | **Valid** | v6 runs postprocess and final validation inside `run_all`, then writes the manifest. |
| First-cycle interest omitted on pre-period debt | **Valid** | Condition required a prior statement. v6 accrues from the first cycle. |
| Instalments ignore closure/charge-off | **Valid** | v6 cancels post-closure and post-charge-off instalments and restates plan balances. |
| Raw voucher/activity independently sampled | **Valid** | Verified 0 orphans after the fix (`card_voucher_activity` is derived from the voucher sample). |
| Raw purchase activity uses plan ID | **Valid** | v6 uses the originating transaction ID; 0 orphans. |
| Refund lineage missing | **Valid** | v6 adds `original_transaction_id`; 0 refunds without a parent. |
| Checklist wording (`PAID_FULL` “exceed” vs “short of”) | **Valid** | Corrected in `hand_off_v6.md` §13. |

**Result: 10 / 10 valid.** No overstatement found.

### 2.2 Codex

| Finding | Validity | Evidence |
|---|---|---|
| International classification not implemented | **Valid** | A prior edit had silently failed; v6 derives the flag from merchant country. 0 mismatches measured. |
| “Churn closes every account” false (null churn date) | **Valid** | v6 keeps such customers active instead of marking them churned; 0 churned customers without a date. |
| Instalments form an unreconciled parallel ledger | **Valid** | v6 removes financed principal from revolving billing and posts repayments as cash with `fact_installment_allocation`. Cash reconciles to 0.00. |
| Fraud decision/loss semantics inconsistent | **Valid** | v6 decides fraud before settlement: blocked ⇒ declined, zero loss. 0 blocked losses measured. |
| Build gate does not certify final data | **Valid** | Single fail-closed path + manifest. |
| Refund/reversal lineage | **Valid** | Refunds fixed; **reversals remain unlinked** (accepted limitation, §5). |
| Card-expiry coverage vacuous | **Valid** | Expiry is now enforced per transaction and the final card is renewed only to period end. |
| Credit-limit decreases absent | **Valid** | v6 implements risk-driven decreases with history. |
| `utilization_pct` naming and capping | **Valid observation, recommendation not adopted** | The field is a signed ratio capped at 1.5. Renaming would break v6 consumers; semantics are now documented explicitly in the data dictionary. |
| Payment-state coverage thin (`RETURNED`) | **Valid** | Still absent — accepted limitation (§5). |
| DPD sparse by construction | **Valid** | Statement-date observation, interval-censored. Documented; no fix possible without daily simulation. |
| Raw-layer connectedness partial | **Valid** | Plans, schedules, vouchers and voucher activity now resolve; other raw tables remain illustrative. |

**Result: 12 / 12 valid.** One recommendation (renaming `utilization_pct`) was
declined with rationale.

### 2.3 MiMo

| Finding | Validity | Evidence |
|---|---|---|
| Fraud loss 10–20× industry | **Valid** | v4 measured 1.175% gross / 0.854% net of volume. v6 measures **6.94 bps** net, inside the 5–7 bps industry band. |
| `auth_code` stored as float64, 12× reuse | **Valid** | v6 emits zero-padded six-character strings; 0 malformed approvals. |
| Dangling `dispute_id` (114k) | **Valid** | Column removed; disputes are not modelled. |
| `CREDIT_BALANCE` unvalidated | **Valid** | v6 validates the status mapping and `dim_account.credit_balance`; 0 mismatches. |
| Duplicate emails/phones | **Valid** | v6: 0 duplicate emails, 0 duplicate phones across 50,000 customers. |
| Validator blind spots (minimum due, tolerances, `applied_statement_id`) | **Valid** | v6 tightens tolerances to $0.01 and makes `applied_statement_id` the first allocation; 0 mismatches. |
| Documentation arithmetic | **Valid** | `hand_off_v6.md` §11 separates star-schema and total row counts; README rewritten. |

**Result: 7 / 7 valid.**

### 2.4 Summary of the audit

| Reviewer | Claims tested | Valid | Partially valid | Invalid |
|---|---:|---:|---:|---:|
| Kimi | 10 | 10 | 0 | 0 |
| Codex | 12 | 12 | 0 | 0 |
| MiMo | 7 | 7 | 0 | 0 |
| **Total** | **29** | **29** | **0** | **0** |

**No material claim from any prior review was spurious.** Two were remediated
differently than recommended (`utilization_pct` renamed vs documented;
over-allocation measured against billed rather than closing), both with written
rationale in the generator and documentation.

---

## 3. Independent measurements on the released artifact

Script does **not** import `build/` or the validator.

| # | Check | Result |
|---|---|---|
| 1 | Statement purchases vs approved non-financed transactions | 0.00 |
| 2 | `paid_amount` vs allocated payments | 0.00 |
| 3 | `allocations + unapplied = payment` (card) | 0.00 |
| 4 | Instalment cash vs schedule paid amount | 0.00 |
| 5 | `PAID_FULL` short of closing balance | 0 rows |
| 6 | Schedule `date_key` missing from `dim_date` | 0 rows |
| 7 | Transactions missing merchant geography | 0 |
| 8 | International flag vs merchant country | 0 mismatches |
| 9 | Raw schedule orphans | 0 |
| 10 | Raw voucher-activity orphans | 0 |
| 11 | Payments after account closure | 0 |
| 12 | Cards issued after account closure | 0 |
| 13 | Refunds without a parent transaction | 0 |
| 14 | `credit_balance` vs negative `current_balance` | 0.00 |
| 15 | `CREDIT_BALANCE` status vs negative closing | 0 mismatches |
| 16 | Delinquency stage vs 60/180 policy | 0 mismatches |
| 17 | Blocked fraud attempts carrying loss | 0 |
| 18 | Fraud alert ↔ transaction linkage | 0 missing |
| 19 | `lifetime_purchases` vs approved non-financed purchases | 0.00 |
| 20 | `lifetime_payments` vs card payments | 0.00 |
| 21 | Statement opening = prior closing (excluding the pre-period opening on the first statement) | 0.00 |
| 22 | `payments_amount` vs card cash received in the cycle | 0.00 |
| 23 | Schedule account vs parent plan account | 0 mismatches |
| 24 | Auth codes: six characters, null on declines | 0 violations |

Measured scale: 12,730,758 transactions; 22,047,420 star-schema rows; 27,581,940
rows across 38 tables; 1,768.6 MB; approved spend $1,712.5M; net fraud loss
6.94 bps; 30+ DPD 1.55%; 60+ DPD 0.29%; charge-offs 4.

---

## 4. Material findings in this review

**No release blockers.** The following are recorded as accepted characteristics,
not defects.

| Item | Why accepted |
|---|---|
| `utilization_pct` is signed and capped at 1.5 | Negative values correctly represent accounts in credit. Documented in the data dictionary; renaming deferred to avoid breaking v6 consumers. |
| `dispute_id` absent | Disputes/chargebacks are not modelled. Removing the column was preferred to dangling references. |
| Reversals not lineage-linked | Refunds are; reversals occur in-cycle and are not linked. Documented. |
| Pre-period opening balances are drawn, not simulated | 36,234 first statements start with a balance; continuity holds from the second statement onward. |
| Charge-offs are rare (4) | The 60/180 policy and cancellation logic are exercised but thin. |

---

## 5. Accepted limitations (carried, disclosed)

1. **No causal structure** — limits affect approved spend mechanically; λ is
   untouched; no treatment assignment.
2. **Fraud label is separable by construction** — loss is calibrated, but the
   label derives from a score.
3. **`RETURNED` payments are not exercised.**
4. **DPD observed at statement dates only** (interval-censored).
5. **No collections/recovery cash flows.**
6. **Reversals not lineage-linked.**
7. **Disputes/chargebacks not modelled.**
8. **Raw layer is a connected sample** (~1.2% of transactions).
9. **No prior-year history.**
10. **App events loosely coupled** to transactions.
11. **Identifiers are synthetic but not reserved-range.**

These are disclosed in `hand_off_v6.md` §12 and `README.md` §6.

---

## 6. Documentation corrections made before this record

| Item | Action |
|---|---|
| `README.md` was mojibake from earlier PowerShell writes | rewritten cleanly for v6 |
| `REVIEW_GUIDE.md` said “8 dimensions, 9 facts” | corrected to 8 / 11 |
| `data_dictionary.md` described `utilization_pct` as an unsigned percentage | now states ratio, signed, capped |
| `docs` regenerated from the released Parquet | `refresh_docs` exit 0 |
| `HANDOFF_FOR_REVIEW.md` (v4 era) | marked superseded, content preserved |
| `v5_review_addendum.md` | marked superseded, content preserved |

No earlier reviewer section was modified.

---

## 7. Residual risk

A reviewer could still reject v6 for reasons of **scope**, not correctness:

- requiring `RETURNED` payment states → needs generator work and a rebuild;
- requiring a full (unsampled) raw mirror → size and runtime increase;
- requiring an identified causal design → requires a v7 DGP with known effects;
- requiring a credible ML benchmark → fraud hardening (label noise, regime shift,
  out-of-time/entity splits).

None of these invalidate the released ledger; each is a defined next-version task.

---

## 8. Reproduction

```powershell
$env:CC_OUT_DIR="credit_card_data_v6"
python -m build.run_all          # exits non-zero on any FAIL
python -m build.refresh_docs     # regenerates validation + dictionary from Parquet
```

Independent checks: `hand_off_v6.md` §14, or the script summarised in §3 above.
Release integrity: `credit_card_data_v6/release_manifest.json` (38 file hashes,
8 doc hashes, `released: true`).

---

*End of v6 review record. Later reviewers must append a separately attributed
section; this section must remain verbatim.*
