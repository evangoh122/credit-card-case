"""Static documentation: delinquency lifecycle extension point."""

TEXT = """# Delinquency lifecycle extension (default, cure, roll rates)

**Status: documented extension point, not yet implemented.**

The current generator models delinquency *state* but not its *transitions*.
`fact_delinquency_snapshot` records days-past-due (DPD) and a bucket for every
account at every statement date, and `fact_credit_limit_event` reacts to arrears,
but there is no explicit cure / roll / default / recovery event stream.

This document specifies the logic so it can be implemented consistently by another
model or engineer.

---

## 1. What already exists

| Object | Grain | Relevant fields |
|---|---|---|
| `fact_delinquency_snapshot` | account × statement date | `days_past_due`, `delinquency_bucket`, `balance`, `amount_past_due`, `unpaid_statement_count` |
| `fact_statement` | account × statement cycle | `minimum_due`, `closing_balance`, `paid_amount`, `payment_status`, `is_late` |
| `fact_payment` | one payment | `amount_usd`, `payment_date`, `applied_statement_id` |
| `dim_account` | account | `days_past_due`, `delinquency_bucket`, `account_status` |

Buckets already used: `CURRENT`, `1-29`, `30-59`, `60-89`, `90-119`, `120+`.
Charge-off currently triggers mechanically at 180 DPD inside `build/facts.py`
(`if dpd >= 180: charged_off = True`), with no recovery cash flow.

---

## 2. Where to implement

In `build/facts.py`, inside `simulate()`, in the statement-cycle loop, immediately
after the delinquency snapshot is appended:

```python
# current code
overdue_now = [...]
dpd = ...
delinq.add(snapshot_id=..., days_past_due=dpd, ...)
if dpd >= 180:
    charged_off = True
```

Add a transition block there. The loop already knows, for each account-cycle:

- `dpd` and the previous cycle's `dpd` (carry a `prev_dpd` variable)
- `amt_past` (unpaid minimum)
- `closing` (total balance)
- `paid_this_cycle` (cure signal)
- `overdue` (list of unpaid statements)

---

## 3. Proposed model

### 3.1 Cure

A **cure** occurs when an account in a delinquent bucket returns to `CURRENT`,
i.e. all past-due minimums are cleared.

```
cured = (prev_dpd > 0) and (dpd == 0)
```

Cure probability should fall with bucket depth. Suggested base rates per month:

| From bucket | Cure rate |
|---|---:|
| 1-29 | 0.55 |
| 30-59 | 0.38 |
| 60-89 | 0.24 |
| 90-119 | 0.15 |
| 120+ | 0.08 |

These interact with the existing payment behaviour: an account that pays in full
cures automatically. Do **not** draw an independent cure on top of a payment that
already clears the balance — apply cure probability only to accounts that remain
past due after the scheduled payment.

### 3.2 Roll (deterioration)

A **roll** is movement to a deeper bucket without curing.

```
rolled = bucket_rank(dpd) > bucket_rank(prev_dpd)
```

Rather than drawing rolls independently, they emerge from missed minimums. If you
want explicit roll-rate calibration, the standard matrix is:

| From | Stay | Roll | Cure |
|---|---:|---:|---:|
| CURRENT | 0.972 | 0.028 | - |
| 1-29 | 0.61 | 0.16 | 0.23 |
| 30-59 | 0.55 | 0.20 | 0.25 |
| 60-89 | 0.50 | 0.24 | 0.26 |
| 90-119 | 0.45 | 0.40 | 0.15 |
| 120+ | 0.42 | 0.50 | 0.08 |

Use either the emergent approach (simplest, current architecture) or the explicit
matrix (more controllable). Do not use both.

### 3.3 Default and charge-off

```
DEFAULT_DPD      = 180      # when the account is charged off
CURE_WINDOW_DAYS = 30       # a payment after charge-off can still reinstate
```

On charge-off:

- set `dim_account.account_status = 'CHARGED_OFF'`
- set `closed_date`
- stop generating new transactions (the simulator already breaks the cycle loop)
- record a charge-off event with `balance_at_chargeoff`
- open a recovery stream (see §3.4)

### 3.4 Recovery

Add `fact_recovery` (or a `recovery_amount` on the charge-off event):

| Column | Type | Notes |
|---|---|---|
| `recovery_id` | string | PK |
| `account_id` | string | FK |
| `chargeoff_date` | date | |
| `recovery_date` | date | |
| `recovery_amount` | decimal | USD |
| `recovery_channel` | string | `COLLECTIONS_AGENCY`, `LEGAL`, `SALE_OF_DEBT`, `CUSTOMER_PAYMENT` |
| `cumulative_recovery_pct` | decimal | recovered / balance at charge-off |

Typical recovery assumptions for a prime US card book: 15-25% of charged-off
balance over 24 months, front-loaded (about 40% of total recovery in the first six
months).

### 3.5 Proposed transition table

`fact_delinquency_transition` — one row per bucket change:

| Column | Type | Notes |
|---|---|---|
| `transition_id` | string | PK |
| `account_id` | string | FK |
| `transition_date` | date | statement date of the change |
| `date_key` | int | FK to `dim_date` |
| `from_bucket` | string | `CURRENT` / `1-29` / ... |
| `to_bucket` | string | |
| `transition_type` | string | `CURE`, `ROLL`, `STAY`, `CHARGE_OFF`, `REINSTATE` |
| `days_past_due_before` | int | |
| `days_past_due_after` | int | |
| `balance` | decimal | balance at transition |
| `amount_past_due` | decimal | |
| `months_in_bucket` | int | |
| `vintage_month` | string | account open month (for cohort analysis) |

This table makes roll-rate, cure-rate and vintage curves directly queryable
without reconstructing transitions from snapshots.

---

## 4. Config knobs to add

Add to `build/config.py` (placeholders already reserved there):

```python
DELINQUENCY_LIFECYCLE_ENABLED = False
DEFAULT_DPD        = 180
CURE_WINDOW_DAYS   = 30
ROLL_MATRIX        = {...}   # bucket -> (stay, roll, cure)
RECOVERY_CURVE     = {...}   # months_since_chargeoff -> cumulative recovery %
```

Keep `DELINQUENCY_LIFECYCLE_ENABLED = False` until the logic is implemented, so the
current build stays reproducible.

---

## 5. Validation to add

When implemented, extend `build/validation.py` with:

- `30+ DPD share` within a target band (currently 1-3% for this book)
- roll-rate matrix by bucket matches the configured rates within tolerance
- cure rate by bucket decreases monotonically with depth
- charged-off accounts never generate transactions after `closed_date`
- `sum(recovery) / sum(balance_at_chargeoff)` within the target band (15-25%)
- every `fact_delinquency_transition` row resolves to an account and a date

---

## 6. Notes for the implementer

- The simulator is **deterministic** (`SEED` in `config.py`) and derives an
  independent RNG stream per step via `utils.rng_for()`. Add a new stream
  (`rng_for("delinquency")`) rather than reusing an existing one, or downstream
  tables will shift.
- Account balances already reconcile exactly:
  `current_balance = last statement closing - later payments`. Any lifecycle logic
  that writes off balances must preserve that identity or explicitly break it in a
  documented way.
- `fact_statement.payment_status = 'UNPAID'` is the natural trigger for a roll.
- The raw mirror (`raw/card_loan_account`, `raw/card_purchase_schedule`) also
  carries arrears fields; keep them consistent if you change the star schema.
"""

CONFIG_NOTE = """
# --------------------------------------------------------------------------
# Delinquency lifecycle (reserved for a later build)
# --------------------------------------------------------------------------
# See docs/delinquency_extension.md. These values are inert until
# DELINQUENCY_LIFECYCLE_ENABLED is set to True and the transition block is added
# to build/facts.py.
DELINQUENCY_LIFECYCLE_ENABLED = False
DEFAULT_DPD = 180
CURE_WINDOW_DAYS = 30
RECOVERY_TARGET_PCT = (0.15, 0.25)
"""
