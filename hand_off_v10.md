# Hand-off v10 — session plan

> **STATUS: PLANNED — NO v10 ARTIFACT EXISTS.**
> v7, v8, v9 and v10 are all unrealised plans. The current released dataset is
> **v6** (`credit_card_data_v6/`). This document must not be cited as a release
> record, and no v10 build, manifest, measurement, or release status is claimed
> here.

**Hand-off author:** MiMo
**Reviewing model:** MiMo-V2.6-Pro (Xiaomi MiMo V2.6 Pro — provider
`xiaomi-token-plan-sgp`, model ID `mimo-v2.6-pro`)
**Date:** 2026-10-25, ~22:20 +08:00 (Asia/Singapore)
**Matching reviews:** `feedback_v8.md` (MiMo section), `feedback_v7.md`
**Evidence basis:** this hand-off contains **no v10 measurements**. Every v6
figure below is a measurement of the v6 artifact cited to its source; everything
else is proposal-tier and labelled as such.

---

## 1. Current state

| Version | State | Evidence |
|---|---|---|
| v6 | released at the built-in gate: exit 0 · 143 PASS / 17 INFO / 0 FAIL · `released: true`; **disputed** — settlement cash reuse (measured) and two documentation hashes drift from the manifest (measured) | `credit_card_data_v6/release_manifest.json`, `feedback_v7.md`, `feedback_v8.md` |
| v7 | planned; no artifact | `hand_off_v7.md` |
| v8 | planned; no artifact | `hand_off_v8.md` |
| v9 | planned; no artifact | `hand_off_v9.md` |
| v10 | planned; no artifact | this document |

Measured v6 facts quoted above, with sources:

- gate result — `release_manifest.json` and `docs/validation_report.md` (counts
  re-derived directly from the report);
- settlement: **47,367 of 54,369 accounts (87.12%)** over-report settlement;
  aggregate `paid_closing_amount` $2.18B vs card cash $1.60B (**$579M excess**);
  max per-account excess $205,952 — `feedback_v7.md` / `feedback_v8.md` (MiMo);
- documentation integrity: `data_dictionary.csv` and `data_dictionary.md`
  SHA-256 **MISMATCH** `release_manifest.json`; the other 6 documented files
  match — `feedback_v8.md` (Codex, independently reproduced by MiMo);
- statement continuity: `opening(i+1) = closing(i)` holds at $0.00 across 530,594
  rolls; the falsified form `closing(i) - payments(i+1)` shows max residual
  $80,026.51 — `feedback_v7.md` (MiMo).

---

## 2. What changed since `hand_off_v9.md`

| Change | Evidence tier |
|---|---|
| MiMo plan review appended to `feedback_v8.md` (concurring with Codex/Kimi; added findings A1–A5) | document inspection + measurement |
| v6 documentation hash drift independently reproduced and attributed to the post-build `refresh_docs` run recorded in `feedback_v6.md` §6 | direct measurement |
| Settlement defect quantified at scale (figures above) | direct measurement |
| This next-numbered planned hand-off created to continue the chain | documentation only |

No generator, dataset, or prior hand-off was changed.

---

## 3. Scope governance (before any v10 feature is chosen)

The project now carries **three unrealised planned hand-offs** (v7, v8, v9) with
overlapping scope. Before v10 scope is selected, an owner must record for each of
v7, v8, v9 whether it will be **built, cancelled, or absorbed** — proposal only,
no decision is invented here. v10 does not claim their scope.

Inherited blockers (proposal; each must close or be explicitly accepted with
rationale before v10 feature work):

| Item | Acceptance test |
|---|---|
| Settlement semantics resolved | field explicitly defined as cash-consuming **or** non-additive; if cash-consuming: per-account `sum(paid_closing_amount) <= card cash`, each payment consumed at most once, independent reconstruction matches every statement; if non-additive: sums prohibited and `PAID_FULL` not derived from it; the v6 settlement regression rewritten so no test encodes $200 of settlement from $150 of cash |
| Statement continuity | `opening(i+1) = closing(i)`; max absolute residual < $0.01 on the artifact (v6 baseline: $0.00) |
| Manifest integrity | every manifest-listed file matches its recorded SHA-256 **after the last mutating step**; no documentation refresh after manifest write; wrapper verifies hashes before reporting success (v6 baseline: 2 mismatches) |
| Independent check suite | committed script, no `build/` imports, covering the §4 contract; exits nonzero on a seeded defect |
| Fraud-loss gate | `0.0003 <= net_loss / approved_spend <= 0.0012` asserted in the gate (v6 measured 6.85 bps net on approved purchase spend; the validator today only checks `loss >= 0`) |
| `auth_code` decision | enforce uniqueness, or accepted-limitation entry with rationale (v6 measured 10.82× mean reuse, max 33×) |
| Charge-off incidence | if recovery scope is in play: either incidence calibrated so distributional acceptance is meaningful (v6 baseline: 4 charge-offs / 584,963 snapshots), or recovery acceptance restricted to structural identities |
| Complete identity set | non-regression contract covers all corrected v6 identities (not a hand-picked subset) |

---

## 4. Non-regression contract (corrected set)

A future v10 artifact must demonstrate, measured on the artifact:

```text
sum(card allocations per payment) + unapplied_amount = card payment amount
statement paid_amount = sum(statement allocations)
statement paid_amount <= statement billed_amount
opening(i+1) = closing(i)
statement purchases_amount = sum(approved non-financed purchases per statement)
statement refunds_amount = sum(refunds per statement)
sum(financed purchase amounts) = sum(plan.principal_amount)
sum(instalment allocations per schedule) = schedule.paid_amount
transaction.is_international = (merchant.country_code != "US")
transaction.merchant_geo_id = merchant.geo_id
all raw foreign keys resolve; no post-closure activity
all fact dates resolve to dim_date
delinquency stage matches the documented 60/180 DPD thresholds
fraud net loss within 3-12 bps of approved spend
every manifest-listed current file matches its recorded SHA-256
```

Plus, once settlement semantics are approved: the settlement invariant from §3.
Note the `PAID_FULL => paid_closing_amount >= closing_balance` row is
**definitional** (`build/facts.py:1436`) and is not a substitute for cash
conservation. A regression blocks release; do not relax a check to pass.

---

## 5. Required artifacts and commands

No v10 files exist yet. A future v10 candidate must provide:

- `credit_card_data_v10/release_manifest.json` (including file SHA-256 values
  computed after the last mutating step);
- `credit_card_data_v10/RELEASE_OK` present, `RELEASE_FAILED` absent;
- `credit_card_data_v10/docs/validation_report.md` with zero `FAIL`;
- a runnable independent-check script and its results;
- `feedback_v10.md` with actual reviewer/model attribution;
- this hand-off updated with measurements from the v10 artifact.

Wrapper invocation (the wrapper exists; `run_v8.ps1` does not):

```powershell
.\run_release.ps1 -Version 10
```

Do not run a mutating documentation refresh after the manifest is written.

---

## 6. Review contract for `feedback_v10.md`

Per `AGENTS.md`/`CLAUDE.md`: dataset version; reviewer name and exact model
identity (model name plus provider/model ID); review date and timezone; artifacts
inspected; evidence basis (measurement / test / code inspection kept distinct);
verdict and every material finding including accepted limitations and required
remediation. If no independent review was performed, say so. Later reviewers
append separately attributed sections; earlier sections are preserved verbatim.

---

## 7. Definition of done

- [ ] v7, v8, v9 each recorded as built, cancelled, or absorbed
- [ ] §3 inherited blockers closed or explicitly accepted with rationale
- [ ] v10 scope frozen with numeric acceptance thresholds per item
- [ ] build exits 0 and the built-in validation has zero `FAIL`
- [ ] §4 identities independently reproduced from the artifact
- [ ] markers and all current-file hashes verify after the last mutating step
- [ ] `feedback_v10.md` records the actual artifact review with model provenance
- [ ] this hand-off rewritten from *planned* to *released* using v10 measurements
- [ ] all earlier hand-offs and reviewer sections retained (banner if superseded)

Until every item passes, v10 remains **planned**.

---

*No v10 data, manifest, or validation report exists at the time of writing. This
is a plan, and it is labelled as one.*
