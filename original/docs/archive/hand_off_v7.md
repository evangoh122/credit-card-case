# Hand-off v7 — session plan

> **STATUS: PLANNED — NO v7 ARTIFACT EXISTS.**
> The current released dataset is `credit_card_data_v6/`. This document is the
> hand-off for the *next* build. Do not cite it as a release record, and do not
> reuse v6 numbers as if they were v7 measurements.

**Convention:** a new `hand_off_vN.md` is written at the end of every session, and a
matching `feedback_vN.md` is created for every release. The hand-off records the
state and the plan; the feedback file records independent review.

**Author:** DeepSeek V4.1 Flash 1017 (provider deepseek, model ID deepseek-flash)
**Author role:** session hand-off author; not an independent reviewer of v7 (no v7 artifact exists).

---

## 1. Where we are

| Item | State |
|---|---|
| Current release | **v6** (`credit_card_data_v6/`) |
| Gate | exit 0 · 143 PASS / 17 INFO / 0 FAIL · `released: true` |
| Independent checks | 24 / 24 at zero (see `docs/archive/feedback_v6.md` §3) |
| Review record | `docs/archive/feedback_v6.md` — provenance, 29/29 prior findings validated |
| Superseded | v3, v4, v5; `docs/archive/HANDOFF_FOR_REVIEW.md` and `docs/archive/v5_review_addendum.md` are bannered |
| Model that produced v6 | DeepSeek V4.1 Flash 1017 (`deepseek` / `deepseek-flash`) |

### Session history

| Session | Output | Outcome |
|---|---|---|
| 1 | v3 | rejected — stale cycle aggregates, payment allocation, instalments |
| 2 | v4 | rejected — fail gate, settlement semantics, lifecycle |
| 3 | v5 | abandoned mid-build — international/geography/churn findings |
| 4 | **v6** | **released** — ledger identities enforced and independently reproduced |

### 1.1 Evidence basis for the claims above

| Claim | Evidence tier |
|---|---|
| v6 gate: exit 0, 143 PASS / 0 FAIL, `released: true` | **measurement** — `release_manifest.json`, the v6 build log (deleted from the repo; see git history) |
| 24 independent checks at zero | **measurement** — Parquet read directly, script in `docs/archive/feedback_v6.md` §3 |
| 29/29 prior findings valid | **measurement + code inspection** — `docs/archive/feedback_v6.md` §2 |
| v7 scope, acceptance tests, non-regression contract | **proposal** — nothing implemented, no v7 artifact |

No figure in this document is carried over from a different version.

---

## 2. Mandatory reading before starting v7

1. `docs/archive/feedback_v6.md` — what was verified and what was accepted as limitation.
2. `docs/archive/hand_off_v6.md` §5 — the 11 ledger identities that must not regress.
3. `docs/archive/hand_off_v6.md` §12 — the disclosed limitations this version should reduce.
4. `credit_card_data_v6/release_manifest.json` — hashes and gate status.

---

## 3. Proposed v7 scope

Each item has an acceptance test. An item only ships if the test passes on the
completed artifact, measured independently.

### 3.1 Payment lifecycle — `RETURNED` payments

| | |
|---|---|
| Why | Codex and Kimi both flagged that `payment_status` advertises `RETURNED` but the simulator only emits `POSTED`. NSF/returned ACH changes delinquency, fees and cash application. |
| Work | ~1.5% of ACH/debit payments fail after posting; reverse the allocation, restore balance, charge a returned-payment fee, and let the account roll delinquent if the minimum is unmet. |
| Acceptance | Returned payments exist; reversed allocations net to zero; delinquency recomputes from restored balances; no negative cash. |

### 3.2 Daily delinquency observation

| | |
|---|---|
| Why | DPD is only observed at statement dates, so survival/hazard models are interval-censored. |
| Work | Add a daily (or event-driven) `fact_delinquency_event` capturing bucket transitions with as-of dates, without changing the statement ledger. |
| Acceptance | Transition timestamps land between statement dates; roll/cure counts reconcile to the monthly snapshots; no duplicated transitions. |

### 3.3 Collections and recovery

| | |
|---|---|
| Why | Charge-offs currently end the story. Portfolio recovery analytics is impossible. |
| Work | `fact_recovery` with recovery channel, amount, date; 15–25% of charged-off balance over 24 months. |
| Acceptance | Recoveries never precede charge-off; cumulative recovery within the configured band; `sum(recovery) ≤ balance_at_chargeoff`. |

### 3.4 Lineage for reversals and disputes

| | |
|---|---|
| Why | Refunds are linked (`original_transaction_id`); reversals are not, and disputes are unmodelled. |
| Work | Link reversals; add `fact_dispute` with lifecycle status and outcome. |
| Acceptance | Every reversal resolves to a parent; every dispute resolves to an alert and a transaction. |

### 3.5 ML-grade fraud dataset (separate from the analytics book)

| | |
|---|---|
| Why | Fraud is separable by construction; AUC will be high regardless of the calibrated loss rate. |
| Work | Produce `ml/fraud_*` extracts with label noise (15% indistinguishable fraud, 2% false flags), an out-of-time split, an out-of-entity split, and a fraud-type regime shift in the final quarter. Keep the operational book unchanged. |
| Acceptance | Model PR-AUC is realistic (not ≈1.0); drift detection fires on the regime shift; operational `fact_transaction` is untouched. |

### 3.6 Causal benchmark DGP

| | |
|---|---|
| Why | v6 has no treatment assignment; causal claims are impossible. |
| Work | Add `limit_increase_offer` with known heterogeneous effects, a randomised cashback promo, a pay-over-time offer, and `docs/dgp_manifest.json` recording true parameters. |
| Acceptance | Estimators recover the manifest effects within Monte Carlo error; placebo tests are null. |

### 3.7 Optional / breaking

- Rename `utilization_pct` → `utilization_ratio` (coordinate with consumers).
- Config flag for a **full** raw mirror instead of a sample.
- Daily app-event densification if event-level funnels are required.

---

## 4. Non-regression contract

v7 must keep every v6 identity at zero:

```
sum(alloc) + unapplied = payment
paid_amount = sum(allocations)
paid_amount <= billed_amount
PAID_FULL => paid_closing_amount >= closing_balance
sum(instalment allocations) = schedule.paid_amount
transaction.is_international = (merchant.country_code != "US")
opening(i+1) = closing(i) - payments(i+1)
```

Plus: 0 raw orphans, 0 post-closure activity, 0 date-dimension gaps,
`delinquency_status` consistent with 60/180.

Any regression blocks the release; do not relax the check.

---

## 5. Build and release procedure for v7

```powershell
$env:CC_OUT_DIR="credit_card_data_v7"
python -m build.run_all          # postprocess + final validation are inside
python -m build.refresh_docs     # regenerate docs from the released Parquet
```

1. Build into a **new directory**; never patch a prior release in place.
2. Confirm exit code `0`, `release_manifest.json` → `"released": true`, `RELEASE_OK`.
3. Re-run the independent script in `docs/archive/feedback_v6.md` §3 against v7.
4. Write `docs/archive/feedback_v7.md`: dataset version, reviewer name, model name and
   provider/model ID, date/timezone, artifacts, evidence basis, verdict, findings.
5. Append, never overwrite, prior review sections.
6. Update `docs/archive/hand_off_v7.md` from *planned* to *released* only when steps 1–3 pass.

---

## 6. Decisions needed before v7 starts

| # | Decision | Recommended |
|---|---|---|
| 1 | Which scope items are in v7? | 3.1 + 3.2 + 3.3 (ledger completeness) |
| 2 | Separate ML dataset or columns in the operational book? | separate `ml/` tree |
| 3 | Causal DGP in v7 or a dedicated v8? | v8 — it changes the generator's semantics |
| 4 | Rename `utilization_pct`? | defer unless consumers agree |
| 5 | Full raw mirror? | keep sampled; add a config flag |

---

## 7. Definition of done for v7

- [ ] all in-scope items implemented
- [ ] build exits 0 with `released: true`
- [ ] every §4 identity still zero
- [ ] new acceptance tests pass on the released artifact
- [ ] `docs/archive/feedback_v7.md` written with model provenance and evidence basis
- [ ] `docs/archive/hand_off_v7.md` rewritten from *planned* to *released* with measured numbers
- [ ] README and REVIEW_GUIDE point at v7
- [ ] prior hand-offs and feedback retained, bannered if superseded

---

*No v7 data, manifest, or validation report exists at the time of writing. This is
a plan, and it is labelled as one.*
