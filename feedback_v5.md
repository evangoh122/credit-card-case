# V5 review feedback

**Reviewer:** Codex  
**Review status:** provisional rejection / release candidate only  
**Review time:** 2026-10-25, approximately 13:13 Asia/Singapore  
**Scope:** `v5_review_addendum.md`, current generator and validator code, and the live v5 build state  
**Modification policy:** no generator, data, or existing documentation was modified during review

At review time, v5 had completed fact generation and was still building the raw
mirror. No v5 Parquet files had been written. The addendum was authored before
the build completed, so its statements that defects are "fixed" are implementation
claims rather than verified results from a finished data release.

## Verdict

Do not sign off v5 based on the current addendum or the built-in validation report
alone. Several genuine improvements are present in code, but material defects and
validation gaps remain in international classification, churn closure, installment
cash accounting, fraud decision semantics, final-output validation, documentation,
and build provenance.

## Release-blocking findings

### 1. The international-classification fix is not implemented

The addendum states that `is_international` now derives from merchant geography.
The code does not do this.

- `TxContext` loads `merchant.is_international` into `merch_intl`.
- Transaction generation independently samples `is_intl` from an MCC-specific
  probability.
- `merch_intl` is never used to assign the transaction flag.

The validator checks that merchants and transactions have non-null geography, but
does not compare `fact_transaction.is_international` with the referenced merchant's
country. The prior international-flag inconsistency can therefore survive while all
new geography checks pass.

**Required fix:** derive transaction international status from the selected
merchant's country/geography, then independently verify:

```text
transaction.is_international == (merchant.country_code != "US")
transaction.merchant_geo_id  == merchant.geo_id
```

### 2. "Churn closes every account" remains false

The dimension generator can mark a customer `CHURNED` while leaving `churn_date`
null when the customer onboarded too recently to satisfy the current churn-date
window. Accounts inherit `planned_close_date=churn_date`, so these customers retain
active accounts.

The validator checks transactions after a non-null churn date, but does not require
every churned customer to have a churn date or every one of their accounts to be
closed.

**Required checks:** zero churned customers with null `churn_date`; all accounts of
a churned customer closed on or before that date; no cards, transactions, payments,
or app activity after closure where the model requires it.

### 3. Installments still form an unreconciled parallel ledger

The installment generator assigns `fact_installment_schedule.paid_amount` and
`paid_date` independently. Schedule rows have no `payment_id` and do not connect to
`fact_payment` or `fact_payment_allocation`. At the same time, the originating
purchase remains fully billed through the revolving statement ledger.

This creates two economic views of the same purchase:

1. the full purchase remains in the revolving balance; and
2. the installment schedule independently reports principal, interest, and
   repayments.

The data therefore cannot support reliable installment exposure, profitability,
cash-flow, or delinquency analysis.

**Required fix:** either post installment repayments through `fact_payment` and
reconcile them to the plan, or model a balance conversion that removes the converted
principal from the revolving ledger. Add a payment-to-installment allocation bridge.

### 4. Fraud decision and settlement semantics remain inconsistent

Fraud alerts are built exclusively from transactions already marked `APPROVED`.
The generator then randomly assigns `BLOCK`, `STEP_UP`, or `ALLOW`, and confirmed
cases receive booked losses. Consequently, `BLOCK` decisions can be attached to
settled transactions and generate losses.

The label is also structurally contaminated: the highest-scoring approved
transactions are selected for alerts, and `transaction.is_fraud` is subsequently
defined from confirmed cases in that selected set. Calling the data merely "too
separable" understates the lifecycle inconsistency.

**Required fix:** generate the fraud event before final authorization status. A
successful block should decline or prevent settlement and should not create booked
fraud loss. Preserve false positives and allowed fraud without deriving the label
directly from the feature-selection process.

### 5. The build gate still does not certify the final delivered data

`build.run_all` writes and validates the initial Parquet. `build.postprocess` then
rewrites `fact_transaction` and `dim_card`, and `build.refresh_docs` must be run as a
separate command.

Consequences:

- the build exit gate signs off the pre-postprocess state;
- raw tables are generated before postprocessing and can drift from the final star
  tables;
- a postprocess or refresh failure can leave modified data with a stale report; and
- written output remains present even when validation fails.

**Required fix:** make build, postprocess, final validation, report generation, and
promotion one atomic workflow. Publish the release only after the final validator
returns zero failures.

## High-priority data and modelling gaps

### Refund and reversal lineage

There is no `original_transaction_id` in `fact_transaction`. Raw
`ORIGINAL_PURCHASE_ID` remains an empty placeholder. Refunds and reversals therefore
cannot be audited back to a specific settled purchase.

### Card-expiry coverage is effectively vacuous

Generated expiry years are forced beyond the observation window. The check for
transactions after expiry can pass without exercising a realistic expiry case.
Issue-date and replacement checks remain useful, but expiry behavior is not tested.

### Credit-limit decreases are absent

The simulator emits only `CREDIT_LIMIT_INCREASE`, although project documentation
describes both increases and decreases. Risk-driven decrease behavior and its
account/statement history are not represented.

### Utilization naming and capping

`utilization_pct` is stored as a ratio: `0.25` represents 25 percent. The `_pct`
name is misleading, and statement utilization is capped at `1.5` without that cap
being clearly documented. Negative balances can also produce negative utilization.

### Payment-state coverage is thin

The data dictionary advertises `POSTED`, `RETURNED`, and `PENDING` payment states,
but the simulator emits posted payments. Returned-payment effects on delinquency,
fees, cash application, and account balance are not exercised.

### DPD observations remain sparse by construction

Delinquency is observed at statement dates rather than daily. This naturally
creates a narrow ladder of DPD values. Monthly roll-rate analysis may still be
possible, but daily hazard, cure-time, and survival interpretations require explicit
interval-censoring treatment and disclosure.

### Raw-layer connectedness is only partially established

The new raw schedule-to-purchase sampling fix is useful, but it establishes only
one relationship. Other raw cross-table identifiers, transaction fields, statuses,
and final-star consistency are not comprehensively validated.

## Validation blind spots

The final release validator should independently check all of the following:

- transaction international flag versus merchant country;
- transaction `merchant_geo_id`, merchant name, MCC, and category versus
  `dim_merchant`;
- every churned customer has a churn date;
- every account belonging to a churned customer is closed appropriately;
- installment cash reconciles to `fact_payment` and the revolving balance;
- allocation `account_id` matches both its payment and statement;
- each payment's `applied_statement_id` agrees with its allocation semantics;
- plan account/customer fields match the originating transaction;
- schedule account/customer fields match the parent plan;
- fraud alert account, customer, card, and merchant match the referenced
  transaction;
- statement-to-statement opening/closing balance continuity;
- `statement.payments_amount` reconciles to payments posted during its cycle;
- account lifetime aggregates reconcile to the fact tables;
- limit-event history reconciles to statement and current account limits;
- primary-key integrity for credit-limit events and delinquency snapshots;
- date-key coverage for every dimension and fact date, not only installment dates;
- payment and allocation amounts are positive and allocation dates are valid;
- refund and reversal parent linkage;
- final raw/star reconciliation after postprocessing.

## Addendum defects

### Premature status claims

`v5_review_addendum.md` was written while the build was still running. It should not
state that v5 supersedes v4 or that fixes are complete until a finished artifact has
passed independent validation.

### Missing provenance

The addendum says the generator commit/state is described, but supplies no commit
hash. The repository has no commits and `.gitignore` excludes the generator and
data. The addendum also lacks:

- exact build, postprocess, and refresh commands;
- captured exit codes for each stage;
- source-tree or generator hash;
- dependency-lock hash and Python/platform details;
- build start and completion timestamps;
- Parquet and documentation hashes; and
- a final success manifest tied to the validation report.

### Documentation drift remains

The root `README.md` and `REVIEW_GUIDE.md` still identify v4 as current. The refresh
step only regenerates documentation within the selected output directory. The
configuration comments also still describe a 1:1 PHP-to-USD design, whereas the
active generator uses US-normal ticket calibration.

The statement that all documentation was regenerated from v5 is therefore too
broad.

### Reviewer-checklist wording error

The checklist says `PAID_FULL` should never "exceed" `closing_balance`. The critical
condition is that `paid_closing_amount` must never be **short of** the positive
closing balance. Both lower- and upper-bound identities may be checked, but the
shortfall test is the release-critical one.

## Improvements present in code but not yet proven in released data

The following changes appear to be implemented:

- `dim_date` extends through 2029;
- international geography members exist;
- raw plan and schedule sampling is connected to sampled purchases;
- payment rows contain `unapplied_amount`;
- statements contain `paid_closing_amount`;
- the waterfall caps allocations at incremental billed debt;
- credit balances have explicit fields/statuses;
- validation failure counting propagates to `run_all`;
- charge-off staging uses strictly greater than 180 DPD; and
- post-closure payment and card-issuance guards are present.

These must still be verified against the completed, postprocessed v5 Parquet rather
than accepted from code inspection.

## Required release procedure

1. Complete the build in a clean staging directory.
2. Run postprocessing within the same orchestrated process.
3. Run final validation against the postprocessed files.
4. Run independent checks that do not import the project validator.
5. Correct the international, churn, installment, and fraud blockers above.
6. Update root and output documentation from measured v5 results.
7. Commit or hash the exact generator source and dependency lock.
8. Create a manifest containing commands, timestamps, exit codes, row counts,
   schemas, and SHA-256 hashes.
9. Atomically promote the staging directory only if every required check passes.

Until those conditions are satisfied, v5 should remain a release candidate and
should not be used for financial, installment, fraud, delinquency-survival, or ML
benchmark analysis.

---

## MiMo — outstanding findings (2026-10-25)

*Independent Parquet review of `credit_card_data_v4/` (~90 checks) plus the current
`build/` source. Only items **not already covered** by the reviews above are listed;
blocker confirmations, status tables and checklists are omitted as duplicative.
Figures are measured on v4 (v5 was still in `raw_mirror` with no Parquet written) and
must be re-measured on `credit_card_data_v5/`. No generator or data was modified.*

1. **Fraud loss economics are ~10–20× industry.** Gross fraud loss is **1.175%** of
   approved spend, net **0.854%** ($20.22M gross / $14.70M net on $1.72B approved
   spend), against ~5–7 bps of volume for US card fraud. "Too separable" understates
   this: the loss *totals* are not credible for a US portfolio, independent of the
   label-contamination problem described above.

2. **`fact_transaction.auth_code` is `float64`** — 899,992 distinct values across
   10,891,737 approved rows (~12× reuse), 1,805,086 nulls (declines and reversals),
   while `data_dictionary.md` describes a "Six-digit authorisation code".

3. **114,438 `dispute_id` values on `fact_fraud_alert` have no dispute table.**
   Dangling references; dispute and chargeback analysis cannot be built on this
   schema, and no validator check covers them.

4. **`CREDIT_BALANCE` ships unvalidated.** `build/facts.py:1355` emits the status and
   `build/facts.py:1098` populates `dim_account.credit_balance`, but
   `build/validation.py` never references `CREDIT_BALANCE`. The repair for the 14,628
   negative-closing statements (−$45.58M) therefore has no gate check of its own.

5. **`dim_customer` identity duplicates:** 112 duplicate emails, 11 duplicate phones
   across 50,000 synthetic customers.

6. **Validator gaps not listed under "Validation blind spots":**
   `build/validation.py:251` tests `minimum_due <= closing_balance.clip(lower=0)`, so
   negative-closing statements pass by construction; `:277` measures over-allocation
   against `billed_amount` rather than `closing_balance`; `:284` still allows a
   **+$1.00** tolerance on `paid_closing_amount`; and `:287` checks only that
   `pay.applied_statement_id` names an existing statement — in v4, **6,946 payments**
   carried an `applied_statement_id` that was not among their own allocation rows.

7. **Documentation arithmetic:** `HANDOFF_FOR_REVIEW.md` §1 reports "Total rows
   27,304,639" where its own table sums to 21,022,560 (the total silently includes
   19 raw tables); `REVIEW_GUIDE.md` still reads "8 dimensions, 9 facts"; `README.md`
   cites an SSN "900–999 area range" although no SSN column exists (only
   `ssn_last4` and hashed raw fields).

---

## Kimi — additional v5 review comments

I reviewed the v5 addendum, the current generator, and the build state. At review
time, the v5 run had only reached `START facts`; no completed v5 Parquet or
regenerated output documentation was available. The addendum's statement that v5
was rebuilt and all documentation regenerated was therefore premature. The
findings below are static code and documentation findings, not measurements from
completed v5 output. I would not accept the current run even if it exits 0.

### Critical: statement settlement double-counts allocations

`build/facts.py:1797-1804` calculates statement `i`'s `paid_closing_amount` by
summing allocations to **every statement `j <= i`** after statement `i`'s cut.
An allocation intended for an older statement can consequently settle every newer
statement as well. For example, two $100 statements followed by one $100 payment
allocated to statement 1 after statement 2 cuts can produce `settled = 100` for
both statements. `PAID_FULL` then trusts this derived value at
`build/facts.py:1361-1362`.

This defeats the central v4 shortfall repair. Checking `paid_closing_amount`
against the same derived field, as proposed in `v5_review_addendum.md:134-136`,
cannot detect it. Settlement must be reconstructed independently from payment
allocations and tested with multi-statement cases, including payments made after
later statement cuts.

### High: final payments do not update account delinquency

Statement-time `last_dpd` is captured at `build/facts.py:1724-1743`. Payments
scheduled after the final statement are applied later at
`build/facts.py:1768-1795`, but account state reuses the old value at
`build/facts.py:1834-1839`; `finalize_accounts` carries it into `dim_account`
unchanged. A final payment can therefore cure the underlying delinquency while
leaving the account dimension delinquent. Recompute final DPD after all eligible
payments and add account-level reconciliation checks.

### High: the geography repair creates contradictory domestic merchants

`us_geo` samples the entire location dimension, including international rows
(`build/dimensions.py:110`). Domestic merchants can thus receive an international
`geo_id`, city, and state while being forced to `country_code="US"`
(`build/dimensions.py:124-133`). `is_international` is independently drawn at
`:151`. Null-geography checks will pass these contradictions. Domestic sampling
must be restricted to US locations, and merchant country, joined location
country, transaction flag, and merchant flag must be checked together.

### Additional gaps

- The release gate still validates pre-postprocess artifacts in
  `build/run_all.py:148-170`. An `EXITCODE=0` requirement is insufficient unless
  postprocessing and final validation are orchestrated in the same release path.
- First-cycle interest is omitted on pre-period debt: the condition at
  `build/facts.py:1647-1650` is false when there are no prior statements, even
  when the opening revolving balance should accrue interest.
- Installment schedules ignore account closure and charge-off
  (`build/facts.py:1388-1407`), so independently generated paid dates and plan
  status can contradict the account lifecycle.
- Raw voucher and activity tables independently sample purchases with different
  seeds (`build/raw_mirror.py:367-369` and `:421-424`), leaving voucher activity
  without matching vouchers and vice versa.
- Raw purchase schedules use the originating transaction ID while purchase
  activity uses the plan ID for `PURCHASE_ID`
  (`build/raw_mirror.py:499-502` and `:535-538`), preventing a consistent join.
- Refunds still need original-purchase lineage. Fraud economics/lifecycle and
  the unreconciled parallel installment cash ledger remain analysis and ML
  blockers, as described in the preceding review.

### Reproduction evidence is still incomplete

The addendum's verification queries are useful checks, but they do not reproduce
the build. A release record needs the exact build/postprocess/refresh command
sequence; source, configuration, source-data, and dependency hashes; output
hashes; independent settlement reconstruction; final-account-DPD checks; and a
two-run determinism comparison. These should accompany the completed data, not
be inferred from generator inspection.

### Required before release

1. Fix and independently test cross-statement settlement and final DPD.
2. Enforce merchant/location/international consistency and lifecycle-aware
   installment generation.
3. Repair raw identifier and sampling connectivity.
4. Run postprocessing before final validation in one fail-closed release process.
5. Publish measured results and provenance from the exact final artifacts.
