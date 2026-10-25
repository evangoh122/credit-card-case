# Feedback — dataset v8 (review of the v8 plan)

## Reviewer provenance

| Field | Value |
|---|---|
| Dataset / release version | **v8 — planned; no v8 artifact exists** |
| Reviewed object | `hand_off_v8.md` |
| Reviewer name | **Codex** |
| Reviewing model | OpenAI Codex (GPT-5 family; exact provider model ID is not exposed in this session) |
| Review date and timezone | 2026-10-25, Asia/Singapore |
| Review type | Plan review, code inspection, release-process inspection, current-manifest integrity measurement, and regression-test execution |
| Artifacts inspected | `hand_off_v8.md`, `feedback_v7.md`, `AGENTS.md`, `run_release.ps1`, `build/run_all.py`, `build/facts.py`, `build/validation.py`, `tests/test_v6_regressions.py`, and the current v6 manifest/documentation |
| Evidence basis | Static inspection; current-file SHA-256 recomputation; `python -m pytest -q` |
| Modification policy | No generator code or generated dataset was modified during this review |

> This is the next numbered feedback record after `feedback_v7.md`. Because no
> v8 artifact exists, it reviews the v8 hand-off plan only. A later artifact
> reviewer must append a separately attributed section rather than replacing this
> one.

## Verdict

**V8 is correctly labelled as planned, but the plan is not startable as written.**
It inherits unresolved v7 blockers, repeats two invalid or disputed ledger
identities, names a build script that does not exist, includes a manual release
step that can invalidate manifest hashes, and combines several research programs
without measurable acceptance thresholds.

There is no `credit_card_data_v8/`, v8 manifest, v8 validation report, v8
`RELEASE_OK`, or v8 measurement to assess.

## Plan-level blockers

### P1. V8 inherits features from v7 but not v7's blocking corrections

The v8 plan inherits v7 items A–D, then adds research items E–H. It does not carry
forward the blockers recorded in `feedback_v7.md`:

- resolve cross-statement settlement/cash-reuse semantics;
- correct the statement opening-roll identity;
- add a runnable independent validation suite;
- add the missing fraud-loss bps assertion;
- close or accept the remaining validator and documentation gaps; and
- disclose the split v6 review verdict rather than citing only “24/24” checks.

Returned-payment reversal, delinquency events, recovery, causal effects, and model
benchmarks all depend on an unambiguous balance and cash ledger. Adding them first
would compound rework.

**Required remediation:** add a v8 prerequisite section that imports every open
blocker from `feedback_v7.md`, with an owner and acceptance test. Settlement and
release integrity must be resolved before A–H begin.

### P2. The non-regression contract repeats invalid or disputed identities

The plan repeats:

```text
PAID_FULL => paid_closing_amount >= closing_balance
opening(i+1) = closing(i) - payments(i+1)
```

The first is definitional in the current generator and does not establish that
payment cash is consumed only once. Current generation and validation deliberately
reuse later-payment suffix cash for every earlier statement.

The second identity is false for the current model. The generator rolls the prior
closing balance directly into the next opening balance; cycle payments affect the
next closing calculation. The correct continuity identity is:

```text
opening(i+1) = closing(i)
```

**Required remediation:** replace the opening identity and add a separately
defined, independently reconstructed settlement invariant. If
`paid_closing_amount` remains a non-additive historical outcome measure, say so
and prohibit summing it across statements. If it is meant to allocate cash, make
the algorithm cash-consuming.

### P3. The documented build and release procedure is not executable as written

`hand_off_v8.md` instructs reviewers to run `./run_v8.ps1`, but that file does not
exist. The repository contains `run_release.ps1`, whose default version is 8.

The manual equivalent also runs `python -m build.refresh_docs` after
`python -m build.run_all`. `run_all` already refreshes documentation and then
writes the manifest. A second refresh can mutate files after their hashes are
recorded.

That risk is already observable in v6: current SHA-256 values for
`docs/data_dictionary.csv` and `docs/data_dictionary.md` do not match their
entries in `credit_card_data_v6/release_manifest.json`; the other six documented
files match.

**Required remediation:** cite the real command:

```powershell
.\run_release.ps1 -Version 8
```

Remove the post-manifest documentation refresh, or regenerate and verify the
manifest after any mutating operation. Add a final current-file hash verification
to the wrapper before it reports success.

### P4. The v8 research scope lacks fixed, falsifiable gates

Items E–H combine a causal DGP, an ML fraud benchmark, a 500-draw estimator study,
and four additional statistical analyses. The plan says these tracks must not be
mixed into one build, while the definition of done requires all E–H to be
implemented. The delivery boundary is therefore contradictory.

Several acceptance phrases are not measurable as written:

- “within Monte Carlo error” has no estimator, tolerance, coverage band, or seed;
- “PR-AUC realistic” has no target band tied to label prevalence;
- “drift detection fires” has no test, alpha, power, or minimum effect;
- EVT, survival, Markov, bootstrap, power, and FDR work has no datasets, outputs,
  or pass/fail thresholds.

**Required remediation:** split the operational ledger, causal benchmark, ML
benchmark, and statistics study into independently versioned deliverables or
explicit profiles. Give each a configuration, deterministic seed policy, output
contract, runtime budget, and numeric acceptance thresholds.

## Release-process and review-contract findings

### R1. The plan conflicts with the project's versioned-feedback rule

The hand-off says `feedback_v8.md` is written only after a completed artifact
exists. `AGENTS.md` requires a matching versioned feedback record and explicitly
allows the record to say that no independent artifact review occurred. This file
therefore exists now as a plan review. A future v8 artifact review must append to
it with new provenance.

### R2. `run_release.ps1` does not enforce the full release contract

The wrapper checks the process exit code, manifest existence, and
`manifest.released`. It does not independently verify:

- `RELEASE_OK` present and `RELEASE_FAILED` absent;
- zero `FAIL` rows in the current validation report;
- every current manifest-listed file hash;
- the independent validation suite;
- that the hand-off was updated with artifact measurements.

Its console statement “released with 0 validation failures” is therefore a report
of the built-in gate, not complete independent release certification.

### R3. The independent validation suite is still absent

The v8 plan requires independent checks but provides no executable path. The
results table in `feedback_v6.md` is not a rerunnable script. A versioned script
must read the artifact directly and avoid importing `build.validation` or its
helpers.

### R4. Reviewer assignments are aspirations, not evidence

The MiMo, Codex, and Kimi table may be used as a desired review matrix, but names
must not imply those reviews occurred. Each actual reviewer must record its own
identity, evidence, and verdict when it reviews the artifact.

## Test result

`python -m pytest -q` completed with **8 passed and 8 warnings**. This does not
clear the settlement issue: the settlement test still expects the disputed
cash-reuse behavior.

## What the v8 hand-off gets right

- It clearly labels v8 as planned and states that no artifact exists.
- It distinguishes operational facts from research outputs.
- It requires model attribution and evidence basis for future reviews.
- It requires prior hand-offs and feedback to be preserved.
- It prevents research outputs from silently altering the operational book.

## Required before v8 implementation begins

1. Import and close the blockers and carried-open findings from `feedback_v7.md`.
2. Correct the non-regression identities and define settlement semantics.
3. Provide a runnable independent-check suite.
4. Correct the build command and make the manifest the final mutating release step.
5. Add final marker, validation-report, and hash checks to the release wrapper.
6. Split or profile E–H and add numeric acceptance thresholds.
7. Decide whether v7 will be built, cancelled, or absorbed; record that decision
   without representing either v7 or v8 as released.

---

*End of Codex's v8 plan review. Later reviewers must append separately attributed
sections and preserve this section verbatim.*

---

## Kimi — additional v8 hand-off review

**Dataset/release version:** v8 plan; no v8 artifact exists  
**Reviewer:** Kimi  
**Reviewing model:** Kimi Code CLI; exact provider/model ID not exposed by the CLI  
**Review time:** 2026-10-25 22:04 Asia/Singapore (UTC+08:00)  
**Artifacts inspected:** `hand_off_v8.md`, Codex's v8 review, current settlement
generator/validator, v6 release manifest, and the referenced `run_v8.ps1` path  
**Evidence basis:** document and code inspection; no independent v8 artifact review
was possible because no v8 data, manifest, validation report, or build log exists

### Verdict

I concur with Codex's P1–P4 and R1–R4 findings. V8 is correctly labelled planned,
but it is not startable as written. The current source still reuses suffix cash
(`build/facts.py:1437-1444`; `build/validation.py:75-86`), so the hand-off's v6
baseline and local `PAID_FULL` check do not establish ledger correctness. The
opening-roll identity must be `opening(i+1) = closing(i)`, not an expression that
subtracts the next cycle's payments again.

I also verified that the primary command's referenced `run_v8.ps1` is absent. The
manual sequence retains a post-manifest documentation refresh and must not be used
as a final release procedure. No v8 measurements are asserted in this review.

### Additional implementation constraints

1. **Returned payments need signed lineage rules.** Before implementation, define
   whether returns are negative allocations or separate linked reversal facts.
   Require original-payment lineage, cap reversals at the original posting, and
   specify effects on allocations, unapplied cash, balances, fees, and DPD.
2. **Resolve the v7 dependency explicitly.** V8 currently inherits A–D only if v7
   ships. Record whether v7 will be built, cancelled, or absorbed; do not leave the
   dependency conditional at implementation time.
3. **Keep planning and release reviews distinct.** This planning review satisfies
   the versioned-review convention, but a future completed v8 artifact still needs
   independent Parquet measurements appended under new reviewer provenance.

### Required before implementation

Close Codex's seven required actions, with settlement semantics, independent cash
conservation, correct opening continuity, a reproducible independent-check suite,
and final-file hash verification treated as prerequisites rather than research-scope
acceptance items. V8 must remain planned until those conditions and the final
artifact gate are satisfied.

---

## MiMo — independent v8 plan review

**Dataset/release version:** v8 plan; no v8 artifact exists (v7 and v9 are also
unrealised plans)
**Reviewed object:** `hand_off_v8.md` (as amended 2026-10-25, §7 amendment record)
**Reviewer:** MiMo
**Reviewing model:** MiMo-V2.6-Pro (Xiaomi MiMo V2.6 Pro — provider
`xiaomi-token-plan-sgp`, model ID `mimo-v2.6-pro`)
**Review time:** 2026-10-25, ~22:15 +08:00 (Asia/Singapore)
**Artifacts inspected:** `hand_off_v8.md`, `hand_off_v9.md`, `feedback_v7.md`,
the Codex and Kimi sections above, `run_release.ps1`, `build/run_all.py`,
`build/facts.py`, `build/validation.py`, `credit_card_data_v6/release_manifest.json`
and its `docs/` files
**Evidence basis:** (a) direct measurement (SHA-256 recomputation on the v6
release; prior parquet measurements cited from `feedback_v7.md`), (b) code
inspection, (c) document cross-reference. No v8 measurement exists or is asserted.
**Modification policy:** no generator, data, or other reviewer's feedback
modified; this section is appended only.

### Verdict

I concur with Codex P1–P4 and R1–R4 and Kimi's constraints: **v8 is correctly
labelled planned and is not startable as written.** Below I independently confirm
the two checkable claims Codex and Kimi make, quantify the settlement defect none
of them scaled, and add findings they did not raise.

### Independent confirmations

| Claim | My verification | Result |
|---|---|---|
| Doc-hash drift in v6 (Codex P3) | recomputed SHA-256 of all 8 manifest-listed `docs/` files against `release_manifest.json` | **confirmed**: `data_dictionary.csv` and `data_dictionary.md` **MISMATCH**; the other 6 match |
| `run_v8.ps1` absent; real wrapper is `run_release.ps1` | file listing + `run_release.ps1` inspection (`-Version` default 8) | **confirmed** |
| Post-manifest `refresh_docs` mutates hashed files | cause identified: the drift was introduced by the post-build `refresh_docs` run that `feedback_v6.md` §6 itself records ("docs regenerated from the released Parquet"), logged in `_refresh_v6.log` — the v6 record documents the exact action that broke its manifest integrity, and `hand_off_v8.md` §5 still prescribes that sequence | **confirmed, with attribution** |
| Falsified opening-roll identity (P2) | measured on v6 parquet (see `feedback_v7.md` P2): `opening(i+1) = closing(i)` holds at **$0.00** across 530,594 rolls; the falsified form shows max residual **$80,026.51**, mean $2,661.91 | **confirmed with scale** |
| Settlement cash reuse (P1) | measured: **47,367 of 54,369 accounts (87.12%)** over-report settlement; aggregate `paid_closing_amount` $2.18B vs card cash $1.60B → **$579M excess (36%)**; max per-account excess $205,952 | **confirmed and quantified** — this is why P1 is a blocker, not a semantics debate |

One narrowing note on `run_release.ps1`: its feedback scaffold is guarded by
`if (-not (Test-Path $fb))`, so it is **append-safe** and will not overwrite an
existing `feedback_v8.md` — the remaining R2 gap is purely that the wrapper
certifies less than it claims (no marker, FAIL-row, or current-hash checks before
its success message).

### Additional findings (MiMo)

| # | Finding | Evidence | Required remediation |
|---|---|---|---|
| A1 | §2.1 item C's acceptance ("cumulative recovery 15–25% over 24 months") would be measured on **4 charge-offs** — the v6 `CHARGE_OFF` stage is effectively empty (4 accounts / 4 snapshots of 584,963, measured). Neither Codex nor Kimi raised the incidence problem | measurement | calibrate charge-off incidence into scope, or make C's acceptance structural (arithmetic identities only) until incidence supports distributional claims |
| A2 | §3's contract is still partial: of the 12 v6 §5 identities it carries 7 (one definitional); missing are statement purchases↔approved non-financed transactions, refunds↔`refunds_amount`, financed purchases↔plan principal, and `transaction.merchant_geo_id = merchant.geo_id` | code + document cross-reference | contract should reference the full corrected identity set (as `hand_off_v9.md` §4 largely does) rather than a hand-picked subset |
| A3 | `auth_code` reuse (measured **10.82× mean, max 33×** over 10.8M approvals) appears in no v7/v8/v9 scope or limitation list, although the v6 audit marks the finding resolved | measurement | decide: enforce uniqueness, or record it as an accepted limitation with rationale (6-digit codes repeat in reality) |
| A4 | §2.2 item H's "acceptance test" cell lists deliverables (EVT, survival, Markov, bootstrap, power, FDR) but no pass/fail threshold at all; §2.2 E/F remain non-falsifiable ("within Monte Carlo error", "PR-AUC realistic") | document cross-reference | numeric gates per the AGENTS.md acceptance-test standard (Codex P4 overlaps; stating the H gap precisely) |
| A5 | §2.1 labels A–D "inherited from v7", but `hand_off_v7.md` numbers them 3.1–3.6 and its decision table recommends only 3.1–3.3 for v7; the mapping is implicit | document cross-reference | cross-reference the item numbers explicitly when the v7/v8 boundary is decided |

### What the amended v8 hand-off gets right

Status labelling and the no-invented-measurements rule hold; the §7 amendment
record keeps the identity correction auditable; §4.2's feedback requirements and
§4.3's aspirational reviewer matrix (with the "each reviewer records its own
identity" rule) match AGENTS.md/CLAUDE.md; and §2's operational-vs-research
separation is the right call — subject to Codex P4's delivery-boundary
contradiction.

### Required before v8 implementation begins (MiMo's combined list)

Codex's 7 actions and Kimi's 3 constraints stand. Add: A1 (charge-off incidence
for recovery), A2 (complete the contract identity set), A3 (auth-code decision),
A4 (numeric gates for E–H, item H in particular).

---

*End of MiMo's v8 plan review. Later reviewers must append separately attributed
sections and preserve this section verbatim.*

