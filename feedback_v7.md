# Feedback — dataset v7 (review of the v7 plan)

**Reviewer:** MiMo
**Reviewing model:** MiMo-V2.6-Pro (Xiaomi MiMo V2.6 Pro — provider
`xiaomi-token-plan-sgp`, model ID `mimo-v2.6-pro`)
**Review status:** plan not startable as written — 2 plan-level blockers, 4 medium findings
**Review time:** 2026-10-25, ~21:50 +08:00 (Asia/Singapore)
**Dataset / release version:** v7 — **`planned`, no v7 artifact exists**. The
reviewed object is `hand_off_v7.md`. The currently released dataset remains v6.
**Scope:** `hand_off_v7.md` (all sections), checked against the AGENTS.md
session-hand-off requirements, the current `feedback_v6.md` record, and direct
measurements on `credit_card_data_v6/`
**Artifacts inspected:** `hand_off_v7.md`; `feedback_v6.md` (current DeepSeek
V4.1 Flash 1017 record); `hand_off_v6.md` §5/§11/§14; `CLAUDE.md`, `AGENTS.md`;
`credit_card_data_v6/` parquet (direct measurement); `build/facts.py`,
`build/validation.py`
**Evidence tiers used:** (a) direct measurement on released v6 parquet, (b) code
inspection, (c) document cross-reference. Each finding below is labelled.
**Modification policy:** no generator, data, hand-off, or other reviewer's
feedback was modified. This file is new and is my review record only.

---

## 1. Verdict

The plan is well-organised and satisfies the AGENTS.md session-hand-off format
(status, evidence tiers, acceptance tests, non-regression contract, commands).
The measured v6 gate figures it quotes are correct (143 PASS / 17 INFO / 0 FAIL —
counted directly in `docs/validation_report.md`). But the plan must not start in
its current form:

1. **The v7 scope omits the settlement cash-reuse remediation** — the one
   release-blocking defect on record from independent review. The plan adds six
   new features and fixes none of the open findings.
2. **The §4 non-regression contract encodes a falsified ledger identity** and is
   therefore unsatisfiable: v7 could not pass its own gate without either
   falsifying data or "relaxing the check" the contract forbids relaxing.

Both are plan-text fixes, not engineering work, and both should land before any
v7 code is written.

---

## 2. Plan-level blockers

### P1. No scope item closes the settlement defect (or any v6 review finding)

Evidence: **measurement + document cross-reference**

`hand_off_v7.md` §3 covers RETURNED payments (3.1), daily delinquency (3.2),
collections (3.3), lineage (3.4), ML fraud (3.5), causal DGP (3.6). §7's
definition of done contains no item resolving open review findings. Meanwhile:

- Two independent reviews on record (Codex, MiMo — both written to
  `feedback_v6.md` earlier this session, since overwritten) **rejected** v6 on
  the cross-statement settlement defect;
- my measurements on the released artifact confirm it is systemic:
  **47,367 of 54,369 accounts (87.12%)** report `sum(paid_closing_amount)` above
  their card cash; aggregate paid_closing $2.18B vs card cash $1.60B → **$579M
  excess (36%)**; max per-account excess $205,952;
- the current `feedback_v6.md` (written by DeepSeek V4.1 Flash 1017, the model
  that produced v6 — a self-review) reports "no release blockers" with a 24-check
  suite that contains **no cash-conservation check**. Its check 5
  (`PAID_FULL` short rows) is tautological: the label is defined as
  `settled >= closing - 0.01` (`build/facts.py:1436`).

Scope items 3.1 and 3.3 *depend* on settlement semantics (reversing allocations,
recomputing delinquency from restored balances, recovery against
balance-at-charge-off). Building them on the defective measure compounds the
rework.

**Required remediation:** add a §3.0 item — cash-consuming settlement
(`_settled_closing_amounts`, `build/facts.py:1408`), each payment consumed once,
documented ordering; rewrite the settlement regression test that currently
encodes the defect (`tests/test_v6_regressions.py:18-32` expects $200 of
settlement from $150 of cash); add the account-level cash-conservation check to
the release gate. Add closure of the remaining v6 findings (below) to §7.

### P2. §4's non-regression contract contains a falsified identity

Evidence: **measurement** (two independent sources)

§4 lists `opening(i+1) = closing(i) - payments(i+1)` and demands it stay "at
zero". Measured on the released v6 data:

- `opening(i+1) = closing(i)` holds **exactly** ($0.00 across 530,594 statement
  rolls) — MiMo measurement;
- the project's own current review record agrees: `feedback_v6.md` §3 check 21 —
  "Statement opening = prior closing (excluding the pre-period opening on the
  first statement) | **0.00**" — DeepSeek measurement;
- the identity as printed in §4 (and in `hand_off_v6.md` §5, from which it was
  copied) has max residual **$80,026.51**, mean $2,661.91.

The generator rolls `balance = closing` (`build/facts.py:1775`); cycle payments
are subtracted once, in `closing(i+1)`. As written, the contract demands a zero
that does not exist in v6 and never did.

**Required remediation:** restate as `opening(i+1) = closing(i)`. Do not copy
`hand_off_v6.md` §5 verbatim — it is wrong on this line (and the correct form is
already proven by two independent measurements).

---

## 3. Medium findings

| # | Finding | Evidence | Required remediation |
|---|---|---|---|
| M1 | §4's contract is a **partial** identity list (7 of the 12 in `hand_off_v6.md` §5) and mixes tautologies with invariants. `PAID_FULL => paid_closing >= closing` is definitional (`facts.py:1436`); the settlement row that *should* be there (cash conservation) is missing | code inspection + measurement | rebuild §4 from the corrected identity set; mark definitional rows as such; include the new cash-conservation invariant |
| M2 | §1 presents v6 as cleanly "released — ledger identities enforced and independently reproduced" and cites "24/24 at zero / 29/29 valid" without disclosing that (a) two independent reviews rejected v6 on settlement, (b) the current `feedback_v6.md` is a **self-review by the builder model** (see its provenance table), and (c) its 24 checks omit cash conservation | document cross-reference | state the split verdict honestly: gate passed; independent review rejected on settlement; the "24/24" suite does not cover it |
| M3 | §5 step 3 says "re-run the independent script in `feedback_v6.md` §3" — that section is a **results table, not a script**; no runnable independent check exists in the repo (`hand_off_v6.md` §14 covers ~5 of the 24) | document cross-reference | commit one independent check script (no `build/` imports) to e.g. `tests/independent_checks.py` covering all contract identities; make §5 step 3 point at it |
| M4 | §3.3 recovery ("15–25% of charged-off balance over 24 months") would be measured on **4 charge-offs** — the v6 `CHARGE_OFF` stage is effectively empty (4 accounts / 4 snapshots of 584,963, measured). Its acceptance test is statistically meaningless at that incidence | measurement | either calibrate charge-off incidence into v7 scope, or state that 3.3's acceptance is structural (arithmetic identities) and cannot yet be distributional |

---

## 4. Minor findings

| # | Finding | Evidence | Required remediation |
|---|---|---|---|
| m1 | §2 says "`hand_off_v6.md` §5 — the **11** ledger identities"; §5 lists **12** | document cross-reference | fix the count |
| m2 | §3.5 acceptance "PR-AUC is realistic (not ≈1.0)" and §3.6 "within Monte Carlo error" are not falsifiable as written; the AGENTS.md standard is an acceptance test | document cross-reference | give thresholds (e.g. PR-AUC band given base rate; recovery tolerance ±X% at N sims) |
| m3 | `auth_code` reuse is recorded as resolved in the v6 audit ("0 malformed approvals" — format only). Measured: **10.82× mean reuse, max 33×** over 10.8M approvals (unchanged from the original finding). §3/§7 do not carry it | measurement + code inspection | add to v7 scope or to accepted limitations with rationale (6-digit codes repeat in reality); do not leave it marked fixed |
| m4 | Fraud-loss figures disagree slightly between "independent" measurements: 6.94 bps on $1,712.5M approved spend (`feedback_v6.md` §3) vs **6.85 bps** on $1,735.8M approved purchase spend (MiMo). Same phenomenon, different denominators; the validator asserts neither (no bps check exists — `build/validation.py:639` is `loss >= 0` only) | measurement | fix one spend definition; add the band assertion `0.0003 ≤ net_loss/spend ≤ 0.0012` to the gate (open v6 finding M1) |
| m5 | §5 runs `python -m build.refresh_docs` after `run_all`, which already runs it | code inspection | harmless; note it is a verification re-run, not a required step |
| m6 | Builder-model provenance (§1 "Model that produced v6 | DeepSeek V4.1 Flash 1017") is asserted in prose; `release_manifest.json` records Python/platform/dependencies but **no builder model**, so it cannot be verified from the artifact | document cross-reference | add a `builder_model` field to the manifest in v7 |
| m7 | `hand_off_v7.md` does not state its own author/model identity (now required by the AGENTS.md session-hand-off rules: "the reviewer or hand-off author name and the reviewing model identity") | document cross-reference | add an author + reviewing-model line to the hand-off header |

---

## 5. Carried-open findings (restated — they were erased from the v6 record)

Note: the `feedback_v6.md` sections containing the Codex and MiMo v6 reviews were
overwritten twice during this session despite the AGENTS.md/CLAUDE.md
preservation rules (the current file even states "No earlier reviewer section was
modified"). Restating the essentials here so the v7 ledger can close them:

1. **Settlement cash conservation** (P1 above) — release-blocking.
2. **Fraud bps assertion missing from the gate** (m4 above) — the calibration is
   fine (measured 6.85–6.94 bps, inside the 3–12 bps band claimed in
   `hand_off_v6.md` §13), the assertion is absent.
3. **Opening-roll identity wrong in docs** (P2 above).
4. **PK coverage gap** — the validator's PK loop (15 tables, not the "14" claimed
   in `hand_off_v6.md` §10) omits `fact_credit_limit_event` and
   `fact_delinquency_snapshot`; snapshots are checked only as
   `len == len(statements)` (`build/validation.py:630`).
5. **`applied_statement_id` validated as set membership**, not as "first
   allocation" as claimed (`build/validation.py:566-571` vs
   `build/facts.py:1645/1907`).
6. **Doc-arithmetic nits** — "8 dims / 10 facts" is 8 / 11; dead `dispute_id`
   entry in `dictionary.py:209`; dead `annotate_statements` with superseded
   settlement semantics (`build/facts.py:1290`); SSN "900-999" claim still in
   `run_all.py:82`; `dim_product`/`dim_channel` outside PK and row-count loops.

None of 2–6 block a rebuild, but the AGENTS.md standard is that a hand-off closes
or explicitly accepts each; §7's definition of done should say so.

---

## 6. What the plan gets right (verified)

- Status labelling (`planned`), evidence-tier table, and the "no figure carried
  over from a different version" claim hold up — I checked the §1 numbers against
  the v6 manifest and validation report: exit 0, 143 PASS / 17 INFO / 0 FAIL,
  `released: true`, 38 tables / 27,581,940 rows are all correct as v6
  measurements (size "1,768.6 MB" is the manifest byte sum in MiB — see the
  naming nit carried from v6).
- `HANDOFF_FOR_REVIEW.md` and `v5_review_addendum.md` are indeed bannered
  `SUPERSEDED` with content preserved — verified.
- Every §3 scope item has an acceptance test, the decision table has
  recommendations, and §5's build procedure correctly requires a clean output
  directory and measured gates before flipping the hand-off to `released`.
- The §3.1 and §3.4 acceptance tests are concrete and falsifiable as written.

---

## 7. Required before v7 starts

1. Add the settlement remediation as a scope item with acceptance tests (P1),
   including the rewritten regression test and the account-level conservation
   check in the gate.
2. Fix the §4 identity to `opening(i+1) = closing(i)` and rebuild the contract
   from the corrected identity set plus cash conservation (P2, M1).
3. Correct §1's release narrative to disclose the split verdict on v6 (M2).
4. Commit a runnable independent-check script; point §5 at it (M3).
5. Decide charge-off incidence for 3.3 (M4) and the falsifiable thresholds for
   3.5/3.6 (m2).
6. Carry the restated open findings (§5) into §3 or §7 explicitly.

With those changes the plan is a good basis for v7. Its structure is not the
problem; its gap is that it plans new features while the release-blocking defect
of the current release goes unfixed.

---

## 8. Post-review actions taken

Per the AGENTS.md forward-hand-off rule (review `hand_off_vN.md` → ensure
`hand_off_v(N+1).md` exists as `planned`): `hand_off_v8.md` already existed
(authored by DeepSeek V4.1 Flash 1017), so it was **amended, not replaced** —
the falsified roll identity was corrected in the non-regression contract, scope
item S (settlement cash conservation) was added, and the v6 release row was
caveated. Amendments are attributed in `hand_off_v8.md` §7. No build, manifest,
measurement, or release status was invented; v7 and v8 remain `planned`.

Finding m7 (hand-off author/model identity) is satisfied by `hand_off_v8.md`'s
author line and the amendment record; `hand_off_v7.md` still needs the line
added by its author.

---

*End of v7 plan review. Later reviewers must append a separately attributed
section; this section must remain verbatim.*
