# Hand-off v8 — session plan

> **STATUS: PLANNED — NO v8 ARTIFACT EXISTS.**
> There is no `credit_card_data_v8/`, no v8 manifest and no v8 validation report.
> The current released dataset is **v6**. This document is the hand-off for the
> next build and must not be cited as a release record.

**Author:** DeepSeek V4.1 Flash 1017 (provider `deepseek`, model ID `deepseek-flash`)
**Author role:** session hand-off author. Not an independent reviewer of v8.
**Date:** 2026-10-25 (build-host local time; release timestamps are UTC)
**Evidence basis:** this document contains **no v8 measurements**. Every v6 figure
quoted below is taken from `credit_card_data_v6/release_manifest.json` and
`docs/archive/feedback_v6.md` §3.

---

## 1. State of the project

| Version | Status | Evidence |
|---|---|---|
| v6 | **released (gate); rejected in independent review on settlement** | exit 0 · 143 PASS / 17 INFO / 0 FAIL · `released: true` · 24/24 independent checks at zero, but that suite has **no settlement cash-conservation check** — see `docs/archive/feedback_v7.md` P1 |
| v7 | **planned** — no artifact | `docs/archive/hand_off_v7.md` |
| v8 | **planned** — no artifact | this document |

v7 and v8 are **both unrealised**. v7 defines ledger-completeness scope; v8 adds
the research track that changes generator semantics. Neither has been built.

---

## 2. What v8 is for

v6 is a credible analytics dataset. Two categories of work remain and they must not
be mixed into one build, because one is additive and the other changes the data
generating process.

### 2.1 Inherited from v7 (additive, ledger-completeness)

| # | Item | Acceptance test |
|---|---|---|
| S | **Settlement cash conservation** — release blocker carried from the v6 reviews (`docs/archive/feedback_v7.md` P1); must ship before E–H | each payment consumed at most once; per-account `sum(paid_closing_amount) <= card cash`; settlement regression test rewritten (the v6 test encodes $200 of settlement from $150 of cash); account-level conservation check in the release gate |
| A | `RETURNED` payments | returned cash reverses its allocation, restores balance, charges a fee; delinquency recomputes; no negative cash |
| B | Daily delinquency events | transitions fall between statement dates; roll/cure counts reconcile to snapshots |
| C | Collections and recovery | `sum(recovery) <= balance_at_chargeoff`; cumulative recovery 15–25% over 24 months; never before charge-off |
| D | Reversal lineage + `fact_dispute` | every reversal and dispute resolves to a parent transaction |

If v7 ships A–D, v8 inherits them and must keep their tests green.

### 2.2 New in v8 (DGP change — research track)

| # | Item | Acceptance test |
|---|---|---|
| E | Causal benchmark DGP | `limit_increase_offer` assigned with documented confounding; known heterogeneous effects in `docs/dgp_manifest.json`; randomised cashback promo; estimators recover truth within Monte Carlo error |
| F | ML-grade fraud extract | separate `ml/` tree with label noise, out-of-time and out-of-entity splits, fraud-type regime shift; operational book untouched; PR-AUC realistic, not ≈1.0 |
| G | Estimator recovery study | 500-draw Monte Carlo reporting bias, RMSE and CI coverage per estimator |
| H | Statistics track | EVT tail index, survival with censoring, roll rates as a Markov chain, bootstrap CIs, power analysis, FDR |

**Do not ship E–H inside the operational fact tables.** E changes how transactions
are generated; it belongs behind a config flag with its own output directory.

---

## 3. Non-regression contract

v8 must keep every v6 identity at zero, measured on the artifact, not the validator:

```
sum(alloc) + unapplied = payment
paid_amount = sum(allocations)
paid_amount <= billed_amount
PAID_FULL => paid_closing_amount >= closing_balance
sum(instalment allocations) = schedule.paid_amount
transaction.is_international = (merchant.country_code != "US")
opening(i+1) = closing(i)
```

> **Corrected 2026-10-25 (MiMo, `docs/archive/feedback_v7.md` P2):** the line above originally
> read `opening(i+1) = closing(i) - payments(i+1)`, copied from `docs/archive/hand_off_v6.md`
> §5. That form is falsified by two independent measurements: `opening(i+1) =
> closing(i)` holds exactly ($0.00 across 530,594 v6 statement rolls; also
> `docs/archive/feedback_v6.md` §3 check 21), while the old form shows max residual $80,026.51.

Once the settlement fix lands (`docs/archive/feedback_v7.md` P1), the contract must also
include, measured per account: `sum(paid_closing_amount) <= card cash`, each
payment consumed at most once. Note the `PAID_FULL` row is definitional
(`build/facts.py:1436`), not an independent invariant.

Plus: 0 raw orphans, 0 post-closure activity, 0 date-dimension gaps, stage mapping
consistent with 60/180 DPD, fraud loss within 3–12 bps.

A regression blocks release. Do not relax a check to pass.

---

## 4. Review contract for v8

`docs/archive/feedback_v8.md` is written **only after** a completed artifact exists.

### 4.1 Required before anyone may call v8 released

| # | Condition |
|---|---|
| 1 | build exit code `0` |
| 2 | `release_manifest.json` → `"released": true` |
| 3 | `RELEASE_OK` present, `RELEASE_FAILED` absent |
| 4 | validation report has zero `FAIL` |
| 5 | independent checks (§3) reproduce against the Parquet |
| 6 | every figure in this hand-off comes from the v8 artifact |

### 4.2 Feedback file requirements

`docs/archive/feedback_v8.md` must record, per `AGENTS.md`:

- dataset/release version;
- reviewer name and exact model identity (model name plus provider/model ID);
- review date and timezone;
- artifacts inspected and evidence basis (measurement / test / code inspection);
- verdict and every material finding, including accepted limitations;
- **if no independent review was performed, say so** rather than leaving the
  fields blank.

Reviewers append separately attributed sections. Earlier sections are preserved
verbatim and never shortened.

### 4.3 Who reviews

| Reviewer | Model identity | Scope |
|---|---|---|
| MiMo | *(to be recorded by the reviewer)* | independent Parquet measurements, economics |
| Codex | *(to be recorded by the reviewer)* | lifecycle, schema, release process |
| Kimi | *(to be recorded by the reviewer)* | settlement, raw connectivity, validator gaps |

The first reviewer to open `docs/archive/feedback_v8.md` owns the provenance header. Later
reviewers append; they do not edit it.

---

## 5. Build and release procedure

```powershell
# one command: build, postprocess, final validation, docs, manifest
.\run_v8.ps1
```

`run_v8.ps1` builds into `credit_card_data_v8/`, and only if the gate passes does
it create `docs/archive/feedback_v8.md` with a provenance header and a **"no independent review
performed yet"** marker. It never invents a verdict.

Manual equivalent:

```powershell
$env:CC_OUT_DIR="credit_card_data_v8"
python -m build.run_all          # exits non-zero on any validation failure
python -m build.refresh_docs     # regenerate docs from the released Parquet
```

---

## 6. Definition of done

- [ ] v7 items A–D either shipped or explicitly deferred with rationale
- [ ] v8 items E–H implemented behind flags with their own outputs
- [ ] build exits 0, `released: true`
- [ ] §3 identities all zero on the v8 artifact
- [ ] E–H acceptance tests pass
- [ ] `docs/archive/feedback_v8.md` written with model provenance and evidence basis
- [ ] this document rewritten from *planned* to *released* with measured numbers
- [ ] README, REVIEW_GUIDE and `docs/archive/hand_off_v7.md` point forward correctly
- [ ] v6 and v7 documents retained; superseded ones bannered, never deleted

---

*No v8 data, manifest or validation report exists at the time of writing. This is a
plan, and it is labelled as one.*

---

## 7. Amendment record

**Amending reviewer:** MiMo
**Reviewing model:** MiMo-V2.6-Pro (provider `xiaomi-token-plan-sgp`, model ID
`mimo-v2.6-pro`)
**Date:** 2026-10-25, ~21:55 +08:00 (Asia/Singapore)

Appended after reviewing `docs/archive/hand_off_v7.md` (record: `docs/archive/feedback_v7.md`), per the
AGENTS.md forward-hand-off rule. The hand-off author remains DeepSeek V4.1 Flash
1017; the following edits were made by the reviewer and are attributed here so
the change history is auditable:

| Edit | Reason |
|---|---|
| §3 contract: `opening(i+1) = closing(i) - payments(i+1)` → `opening(i+1) = closing(i)`, with measurement citation | the old form is falsified — `docs/archive/feedback_v7.md` P2 |
| §2.1: added scope item **S** (settlement cash conservation) | release blocker carried from the v6 reviews — `docs/archive/feedback_v7.md` P1 |
| §1 v6 row: released (gate) / rejected in independent review; caveat on the 24/24 suite | the suite omits settlement cash conservation — `docs/archive/feedback_v7.md` M2 |
| §3: noted the `PAID_FULL` row is definitional; named the conservation invariant to add post-fix | `docs/archive/feedback_v7.md` M1 |

No build, manifest, measurement, or release status was invented; v7 and v8 both
remain `planned`.
