# Hand-off v9 — session plan

> **STATUS: PLANNED — NO v9 ARTIFACT EXISTS.**
> This hand-off was created after review of `hand_off_v8.md`. It does not claim
> that v7, v8, or v9 has been built or released.

**Hand-off author / reviewer:** Codex  
**Reviewing model:** OpenAI Codex (GPT-5 family; exact provider model ID is not exposed in this session)  
**Date:** 2026-10-25, Asia/Singapore  
**Matching review:** `feedback_v8.md`

## 1. Current state

| Version | State | Evidence |
|---|---|---|
| v6 | latest built artifact; built-in gate reported `released: true`, but current directory integrity and settlement semantics remain disputed | v6 manifest, `feedback_v7.md`, `feedback_v8.md` |
| v7 | planned; no artifact | `hand_off_v7.md` |
| v8 | planned; no artifact | `hand_off_v8.md` |
| v9 | planned; no artifact | this document |

The current v6 directory has two documentation files whose present SHA-256 values
do not match the release manifest: `data_dictionary.csv` and
`data_dictionary.md`. No v7, v8, or v9 release measurements exist.

## 2. What changed in this session

| Change | Evidence tier |
|---|---|
| Reviewed the v8 planning hand-off and recorded its blockers | code/document inspection in `feedback_v8.md` |
| Recomputed current v6 documentation hashes; two mismatches remain | direct measurement |
| Executed the current regression suite: 8 passed, 8 warnings | test |
| Created this next-numbered planned hand-off | documentation only |

No generator or dataset was changed.

## 3. Open scope and acceptance tests

V9 scope is not approved. Before selecting new features, the following inherited
work must close:

| Item | Acceptance test |
|---|---|
| Settlement semantics | field is explicitly defined as non-additive or cash-consuming; independent reconstruction matches every statement; regression cases cover payments after later cuts |
| Statement continuity | `opening(i+1) - closing(i)` has maximum absolute residual below $0.01 |
| Independent validation | a committed script runs without importing `build/` and exits nonzero on a seeded defect |
| Release integrity | every current manifest-listed file matches its SHA-256 after all generation and documentation steps |
| Release wrapper | fails on bad markers, report failures, hash mismatches, or independent-check failures |
| Version decision | v7 and v8 are explicitly built, cancelled, or absorbed before v9 claims their scope |
| Research profiles | operational, causal, ML, and statistics outputs have separate configurations and numeric gates |

## 4. Non-regression contract

At minimum, a future v9 artifact must independently demonstrate:

```text
sum(card allocations per payment) + unapplied_amount = card payment amount
statement paid_amount = sum(statement allocations)
statement paid_amount <= statement billed_amount
opening(i+1) = closing(i)
sum(installment allocations per schedule) = schedule paid_amount
transaction international flag = (merchant country != US)
all raw foreign keys resolve
no post-closure activity
all fact dates resolve to dim_date
delinquency stage matches the documented DPD thresholds
every manifest-listed current file matches its recorded SHA-256
```

The resolved settlement invariant must be added once its semantics are approved.
A definitional `PAID_FULL` check is not a substitute for cash conservation.

## 5. Required artifacts and commands

No v9 files exist yet. A future v9 candidate must provide:

- `credit_card_data_v9/release_manifest.json`;
- `credit_card_data_v9/RELEASE_OK` and no `RELEASE_FAILED`;
- `credit_card_data_v9/docs/validation_report.md` with zero failures;
- a runnable independent-check script and results;
- `feedback_v9.md` with actual reviewer/model attribution; and
- this hand-off updated with measurements from the v9 artifact.

Proposed wrapper invocation, after the wrapper satisfies `feedback_v8.md`:

```powershell
.\run_release.ps1 -Version 9
```

Do not run a separate mutating documentation refresh after the manifest is
written.

## 6. Definition of done

- [ ] inherited blockers resolved or explicitly accepted with rationale
- [ ] v9 scope frozen with numeric acceptance thresholds
- [ ] build exits 0 and built-in validation has zero failures
- [ ] independent checks reproduce from the artifact
- [ ] final markers and current-file hashes verify
- [ ] `feedback_v9.md` records the actual artifact review
- [ ] this document is updated from planned to released using v9 measurements
- [ ] earlier hand-offs and reviewer sections remain preserved

Until all items pass, v9 remains **planned**.

