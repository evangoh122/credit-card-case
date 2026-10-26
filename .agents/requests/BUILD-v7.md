# BUILD-v7 — consolidated v7 (coder: DeepSeek flash)

Repo: `/mnt/d/Get shit Done/Data Science/DS Work/Credit Card Case` (branch `master`, baseline `aead184`).
Python: `.venv/bin/python` (3.13). Run everything natively in WSL/Linux; never call `wsl.exe`.
Read first: `AGENTS.md`, `CLAUDE.md`, `docs/archive/hand_off_v10.md` §3–§4, `docs/archive/feedback_v7.md`, `docs/archive/feedback_v8.md`, `docs/archive/hand_off_v7.md` §3.
v8, v9 and v10 are **absorbed into this build**; do not create `credit_card_data_v8..v10`.

## Goal
One new dataset `credit_card_data_v7/` (`CC_OUT_DIR=credit_card_data_v7`) that (A) closes every v6 blocker and
(B) supports the owner's ML/MLOps use cases: fraud classification, credit scoring/default, churn/survival,
RFM+MCC segmentation, anomaly detection, merchant spending forecasting, sequential fraud (test-purchase then big purchase).
Never patch `credit_card_data_v6/` in place.

## Part A — blockers (all required; thresholds are release gates in `build/validation.py`)
1. **Settlement semantics.** Make `paid_closing_amount` cash-consuming: per account `sum(paid_closing_amount) <= card cash`,
   each payment consumed at most once. Rewrite the v6 settlement regression so no test encodes $200 settlement from $150 cash.
   `PAID_FULL` must not be derived from a non-conserving field.
2. **Statement continuity.** `opening(i+1) = closing(i)`, max abs residual < $0.01 (v6 baseline $0.00; do not regress).
3. **Manifest integrity.** Every manifest-listed file matches its SHA-256 *after the last mutating step*. No `refresh_docs` after the
   manifest is written. `run_release.ps1` and a new Linux `run_release.sh` must verify all hashes before reporting success.
4. **Independent check suite.** Committed `checks/independent_checks.py`: no `build/` imports, reads Parquet directly, covers every
   identity in `docs/archive/hand_off_v10.md` §4, exits nonzero on a seeded defect (include `--selftest` that injects a defect and asserts nonzero).
5. **Fraud-loss gate.** Assert `0.0003 <= net_loss / approved_spend <= 0.0012` in the validator (v6 measured 6.85 bps).
6. **auth_code.** Enforce uniqueness per (merchant, day), or write an accepted-limitation entry with rationale in the docs.
7. **Charge-off incidence.** Calibrate so there are >= 400 charge-offs at the default scale, or restrict recovery acceptance to
   structural identities and say so. Do not fake recovery counts.
8. Full non-regression contract = every line of `docs/archive/hand_off_v10.md` §4.

## Part B — ML-grade extracts, under `credit_card_data_v7/ml/` (operational book `fact_transaction` stays untouched)
1. `fraud_*`: label noise (15% indistinguishable fraud, 2% false flags in the ML copy only), out-of-time split, out-of-entity split,
   fraud-type regime shift in the final quarter. Acceptance: a baseline LightGBM/sklearn model gives PR-AUC realistic (< 0.95) and
   the drift check fires on the final-quarter shift. Test-purchase-then-large-purchase sequences present and labelled.
2. `credit_default_*` and `churn_*`: point-in-time features and labels with a documented observation/outcome window; zero label leakage
   (a test proves no feature uses data after the observation date).
3. `segmentation_*` (RFM + MCC share) and `forecast_*` (merchant-month volume with seasonality).
4. `ml/README.md` documenting every table, label definition, split, and known synthetic-data limitation (no claims of real-world calibration).
Keep scope modest: do not build the causal DGP (v8 item) in this build; record it as deferred in the hand-off.

## Rules (from the owner's feedback)
- Tests must not copy production logic. For every new gate add a test that **mutates the production behaviour** and fails
  (seeded defect), and state the mutation in the test docstring. Tests that merely recompute the same formula are rejected.
- Foreground, self-verifying commands only. Do not background tests. Work in small commits on `master`.
- Do not delete or shorten any `feedback_*.md` or `hand_off_*.md`. Do not modify other models' feedback sections.
- Write `docs/archive/hand_off_v7.md` update as `in progress` (never `released` — only the owner's reviewer chain can release). Create `docs/archive/feedback_v7.md`
  section only if it does not break existing content; otherwise append a clearly attributed `DeepSeek (builder, not a reviewer)` section.
- Do not use more than the scale in `build/config.py` unless required for the 400-charge-off gate; record any scale change.

## Required checks before you report
```
.venv/bin/python -m pytest -q -p no:cacheprovider
CC_OUT_DIR=credit_card_data_v7 .venv/bin/python -m build.run_all      # exit 0, released true, zero FAIL
.venv/bin/python checks/independent_checks.py credit_card_data_v7     # exit 0
.venv/bin/python checks/independent_checks.py --selftest              # must exit 0 by catching the seeded defect
```
Record exact outputs in `evidence/BUILD-v7.log`.

## Terminal report (mandatory, last line of your output)
`BUILD DONE | status: <SUCCESS|FAILURE> | sha: <full 40-char SHA or NO_COMMIT> | branch: master | evidence: evidence/BUILD-v7.log`
SUCCESS requires committed changes, SHA == branch HEAD, all checks above actually passing, clean worktree. Otherwise report FAILURE.
