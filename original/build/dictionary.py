"""Data dictionary generation from the materialised DataFrames."""
from __future__ import annotations

import pandas as pd
import polars as pl

from . import config as C

COMMON = {
    "customer_id": "Primary key of the customer (CUST######).",
    "account_id": "Primary key of the credit-card account (ACC#######).",
    "card_id": "Primary key of the physical/virtual card (CRD########).",
    "merchant_id": "Primary key of the merchant (M#####).",
    "geo_id": "Primary key of the location / geography record (LOC#####).",
    "product_id": "Card product identifier.",
    "channel_id": "Transaction channel identifier.",
    "date_key": "Integer date key YYYYMMDD, joins to dim_date.",
    "month_key": "Calendar month YYYY-MM.",
    "first_name": "Customer given name.",
    "last_name": "Customer family name.",
    "full_name": "Customer full name.",
    "date_of_birth": "Date of birth.",
    "age_years": "Age in years at the end of the observation window.",
    "gender": "F / M.",
    "email": "Synthetic e-mail address.",
    "phone": "Synthetic US phone number (E.164).",
    "ssn_last4": "Last four digits of a synthetically generated, non-issuable SSN.",
    "id_type": "Government ID document type used for KYC.",
    "id_number_masked": "Masked government ID number.",
    "home_geo_id": "FK to dim_location for the customer's home address.",
    "address_line1": "Street address line.",
    "city": "City.",
    "state_code": "Two-letter US state code.",
    "zip_code": "US ZIP code (5 digits).",
    "metro": "CBSA metro area name.",
    "cbsa_code": "US Census CBSA code.",
    "region": "US Census region (Northeast, Midwest, South, West).",
    "country_code": "ISO-3166 alpha-2 country code.",
    "latitude": "Latitude of the geography centroid.",
    "longitude": "Longitude of the geography centroid.",
    "timezone": "IANA timezone.",
    "income_annual": "Self-reported annual gross income in USD.",
    "employment_status": "EMPLOYED / SELF_EMPLOYED / RETIRED / STUDENT / UNEMPLOYED.",
    "employer": "Employer name.",
    "occupation": "Occupation description.",
    "industry": "Industry of employment.",
    "education": "Highest education level.",
    "marital_status": "Marital status.",
    "housing_status": "MORTGAGE / RENT / OWN / LIVE_WITH_FAMILY.",
    "credit_score": "Synthetic FICO-style score (300-850).",
    "risk_tier": "PRIME_PLUS / PRIME / NEAR_PRIME / SUBPRIME / DEEP_SUBPRIME.",
    "acquisition_channel": "Marketing channel that acquired the customer.",
    "onboarding_date": "Date the customer completed onboarding.",
    "kyc_status": "VERIFIED / PENDING.",
    "is_email_verified": "Whether the e-mail passed verification.",
    "is_phone_verified": "Whether the phone passed verification.",
    "marketing_opt_in": "Marketing consent flag.",
    "customer_status": "ACTIVE / CHURNED.",
    "churn_date": "Date the customer relationship ended.",
    "customer_segment": "Behavioural segment: AFFLUENT_POWER / PRIME_ACTIVE / CORE_MAINSTREAM / LIGHT_USER / DORMANT.",
    "expected_txn_per_month": "Segment median transactions per month used by the simulator.",
    "product_name": "Commercial product name.",
    "tier": "Product tier (SECURED, STANDARD, SIGNATURE, PLATINUM, GOLD, BUSINESS).",
    "network": "Card network (Visa, Mastercard, American Express).",
    "annual_fee": "Annual fee in USD.",
    "apr_purchase": "Purchase APR (decimal, e.g. 0.2199 = 21.99%).",
    "apr_cash": "Cash-advance APR.",
    "foreign_txn_fee": "Foreign transaction fee rate (decimal).",
    "rewards_rate": "Rewards earn rate per USD spent.",
    "rewards_type": "CASHBACK / POINTS / NONE.",
    "min_credit_score": "Minimum credit score for the product.",
    "min_limit": "Minimum credit limit for the product.",
    "max_limit": "Maximum credit limit for the product.",
    "is_secured": "Secured card flag.",
    "is_business": "Business card flag.",
    "channel_name": "Human-readable channel name.",
    "channel_group": "IN_STORE / ONLINE / MOBILE_APP / PHONE_ORDER / ATM.",
    "entry_mode": "CHIP / CONTACTLESS / SWIPE / ECOM / MOTO.",
    "is_card_present": "True when the card was physically presented.",
    "merchant_name": "Merchant trading name.",
    "merchant_brand": "Merchant brand (without store number).",
    "mcc": "ISO 18245 merchant category code.",
    "mcc_description": "MCC description.",
    "category": "Analytical merchant category group.",
    "channel_primary": "Merchant's dominant channel.",
    "is_recurring": "True for subscription-style merchants.",
    "merchant_size_tier": "SMALL / MID / LARGE.",
    "popularity_weight": "Within-MCC popularity weight used by the simulator.",
    "is_international": "True when the merchant is outside the US.",
    "open_date": "Account opening date.",
    "initial_credit_limit": "Credit limit at account opening.",
    "credit_limit": "Current credit limit.",
    "available_credit": "Credit limit minus current balance.",
    "account_status": "ACTIVE / CLOSED / CHARGED_OFF / FROZEN.",
    "is_primary": "Primary account for the customer.",
    "statement_day": "Day of month on which the statement is cut.",
    "payment_due_days_after_statement": "Grace period in days between statement and due date.",
    "current_balance": "Balance outstanding at the end of the observation window.",
    "utilization_pct": "Balance divided by credit limit, stored as a RATIO (0.25 = 25%). Signed: negative when the account is in credit. Capped at 1.5.",
    "lifetime_purchases": "Gross approved purchase volume over the window.",
    "lifetime_payments": "Total payments posted over the window.",
    "lifetime_interest": "Total interest charged over the window.",
    "lifetime_fees": "Total fees charged over the window.",
    "last_statement_date": "Most recent statement date.",
    "last_statement_balance": "Closing balance on the most recent statement.",
    "last_payment_date": "Most recent posted payment date.",
    "last_payment_amount": "Most recent posted payment amount.",
    "days_past_due": "Days past due at the end of the observation window.",
    "delinquency_bucket": "CURRENT / 1-29 / 30-59 / 60-89 / 90-119 / 120+.",
    "closed_date": "Date the account was closed.",
    "acquisition_cost": "Cost to acquire the account in USD.",
    "card_type": "PHYSICAL / VIRTUAL.",
    "last4": "Last four digits of the card number.",
    "expiry_month": "Card expiry month.",
    "expiry_year": "Card expiry year.",
    "issue_date": "Card issue date.",
    "card_status": "ACTIVE / REPLACED / LOCKED / EXPIRED / CANCELLED.",
    "replacement_reason": "LOST / DAMAGED / FRAUD / None.",
    "is_contactless": "Contactless-enabled card.",
    "is_virtual": "Virtual card flag.",
    "transaction_id": "Primary key of the transaction (TXN#########).",
    "transaction_date": "Authorisation date.",
    "transaction_ts": "Authorisation timestamp.",
    "transaction_type": "PURCHASE / REFUND / CASH_ADVANCE / BALANCE_TRANSFER.",
    "amount_usd": "Signed transaction amount in USD (refunds are negative).",
    "currency_code": "Transaction currency (USD unless cross-border).",
    "original_amount": "Amount in the transaction currency.",
    "fx_rate": "FX rate applied to convert original_amount into USD.",
    "status": "APPROVED / DECLINED / REVERSED.",
    "decline_reason_code": "Numeric/alpha decline reason code.",
    "decline_reason": "Human-readable decline reason.",
    "auth_code": "Six-digit authorisation code.",
    "rewards_points": "Rewards points earned on the transaction.",
    "installment_plan_id": "FK to fact_installment_plan when the purchase was converted to pay-over-time.",
    "fraud_score": "Model fraud propensity score (0-1).",
    "is_fraud": "True when the transaction was confirmed fraudulent.",
    "statement_id": "FK to the statement the transaction rolled into.",
    "source_system": "Originating system for the event.",
    "merchant_geo_id": "FK to the merchant's location.",
    "customer_geo_id": "FK to the customer's home location.",
    "statement_date": "Statement cut date.",
    "cycle_start_date": "First day of the billing cycle.",
    "cycle_end_date": "Last day of the billing cycle.",
    "payment_due_date": "Payment due date.",
    "opening_balance": "Balance carried into the cycle.",
    "purchases_amount": "Approved purchases billed in the cycle.",
    "refunds_amount": "Refunds/credits billed in the cycle (negative).",
    "cash_advances_amount": "Cash advances in the cycle.",
    "payments_amount": "Payments received during the cycle.",
    "interest_charged": "Interest charged on the statement.",
    "fees_charged": "Fees charged on the statement (annual, late, FX).",
    "closing_balance": "Balance at statement cut.",
    "minimum_due": "Minimum payment due.",
    "transaction_count": "Number of authorisation attempts in the cycle.",
    "paid_amount": "Amount paid against this statement.",
    "payment_status": "PAID_FULL / PAID_PARTIAL / PARTIAL / UNPAID / PENDING / NO_BALANCE.",
    "is_late": "True when the payment settled after the due date.",
    "payment_id": "Primary key of the payment (PAY#########).",
    "payment_date": "Date the payment was posted.",
    "payment_method": "ACH / DEBIT_CARD / CHECK / WIRE / INTERNAL_TRANSFER.",
    "payment_channel": "MOBILE_APP / WEB / IVR / BRANCH.",
    "payment_status": "POSTED / RETURNED / PENDING.",
    "is_autopay": "True when enrolled in autopay.",
    "reference_number": "Payment reference number.",
    "applied_statement_id": "Statement the payment was scheduled against.",
    "posted_ts": "Payment posting timestamp.",
    "plan_id": "Primary key of the pay-over-time plan (PLN########).",
    "plan_start_date": "Plan start date.",
    "principal_amount": "Amount converted into the plan.",
    "term_months": "Plan term in months.",
    "promo_apr": "Promotional APR applied to the plan (0 = 0% promo).",
    "total_interest_fee": "Total interest/fee over the life of the plan.",
    "monthly_payment": "Monthly instalment amount.",
    "total_repayment": "Total amount repayable.",
    "remaining_balance": "Remaining balance on the plan.",
    "plan_status": "ACTIVE / COMPLETED / DELINQUENT.",
    "plan_end_date": "Plan end date.",
    "offer_channel": "Where the plan was offered.",
    "schedule_id": "Primary key of the instalment (ISC#########).",
    "installment_number": "Instalment sequence number.",
    "due_date": "Instalment due date.",
    "amount_due": "Instalment amount due.",
    "principal_component": "Principal portion of the instalment.",
    "interest_fee_component": "Interest/fee portion of the instalment.",
    "paid_amount": "Amount paid against the instalment.",
    "paid_date": "Date the instalment was paid.",
    "limit_event_id": "Primary key of the credit limit event (LIM#######).",
    "event_date": "Date of the limit change.",
    "event_type": "CREDIT_LIMIT_INCREASE / CREDIT_LIMIT_DECREASE / TEMPORARY_HOLD.",
    "old_limit": "Limit before the change.",
    "new_limit": "Limit after the change.",
    "change_amount": "Absolute change in limit.",
    "change_pct": "Percentage change in limit.",
    "reason": "Business reason for the change.",
    "channel": "Channel that triggered the change.",
    "snapshot_id": "Primary key of the delinquency snapshot (SNP#########).",
    "snapshot_date": "As-of date (statement date).",
    "amount_past_due": "Total amount past due.",
    "unpaid_statement_count": "Number of statements with an unpaid balance.",
    "alert_id": "Primary key of the fraud alert (FRD#######).",
    "alert_date": "Date the alert was raised.",
    "alert_type": "CARD_NOT_PRESENT / ACCOUNT_TAKEOVER / COUNTERFEIT / LOST_STOLEN / FRIENDLY_FRAUD / APPLICATION_FRAUD.",
    "risk_score": "Alert risk score (0-1).",
    "decision": "BLOCK / STEP_UP / ALLOW.",
    "case_status": "CONFIRMED_FRAUD / FALSE_POSITIVE / CUSTOMER_ERROR / OPEN.",
    "loss_amount": "Gross fraud loss.",
    "recovered_amount": "Amount recovered.",
    "net_loss": "Gross loss minus recovered.",
    "resolution_date": "Case resolution date.",
    "dispute_id": "Dispute identifier.",
    "ip_address": "Originating IP address.",
    "device_id": "Hashed device identifier.",
    "analyst_queue": "TRIAGE / INVESTIGATION / RECOVERY.",
    "event_id": "Primary key of the app event (EVT#########).",
    "allocation_id": "Primary key of the payment allocation (ALC#########).",
    "allocation_date": "Date the allocation was applied (equals the payment date).",
    "session_id": "App session identifier.",
    "event_timestamp": "Event timestamp.",
    "event_date": "Event date.",
    "event_name": "Digital event name.",
    "screen_path": "App screen path.",
    "platform": "ANDROID / IOS / WEB.",
    "device": "Device model or browser.",
    "os_version": "Operating system version.",
    "app_version": "App version.",
    "is_conversion_event": "True for conversion-funnel events.",
    "properties_json": "Raw event properties (JSON).",
    "date": "Calendar date.",
    "year": "Calendar year.",
    "quarter": "Calendar quarter.",
    "month": "Calendar month number.",
    "month_name": "Month name.",
    "month_short": "Three-letter month.",
    "year_month": "YYYY-MM.",
    "day_of_month": "Day number.",
    "day_of_week": "ISO day of week (1=Mon).",
    "day_name": "Weekday name.",
    "day_short": "Three-letter weekday.",
    "iso_week": "ISO week number.",
    "is_weekend": "Weekend flag.",
    "is_holiday": "US federal holiday flag.",
    "is_month_start": "First day of month flag.",
    "days_in_month": "Number of days in the month.",
    "is_month_end": "Last day of month flag.",
    "fiscal_year": "Fiscal year (Oct-Sep).",
    "fiscal_quarter": "Fiscal quarter (Oct-Sep).",
    "product_sk": "Surrogate key for the product dimension.",
    "channel_sk": "Surrogate key for the channel dimension.",
}

FACT_NOTES = {
    "fact_transaction": "Grain: one row per authorisation attempt (approved, declined or reversed). "
                        "Signed amounts: purchases positive, refunds negative.",
    "fact_statement": "Grain: one row per account per monthly statement cycle.",
    "fact_payment": "Grain: one row per posted payment.",
    "fact_payment_allocation": "Grain: one row per payment-to-statement allocation produced by the payment waterfall (oldest due first: minimums, then billed balances).",
    "fact_installment_plan": "Grain: one row per pay-over-time plan.",
    "fact_installment_schedule": "Grain: one row per plan instalment.",
    "fact_credit_limit_event": "Grain: one row per credit limit change.",
    "fact_delinquency_snapshot": "Grain: one row per account per statement date.",
    "fact_fraud_alert": "Grain: one row per fraud alert.",
    "fact_app_event": "Grain: one row per digital analytics event.",
}


def _describe(table: str, col: str) -> str:
    if col == "credit_balance":
        return "Positive amount held as account credit (negative current balance)."
    if col == "paid_closing_amount":
        return "Portion of this statement's closing balance settled by later payments (FIFO view)."
    if col == "payment_status" and table == "fact_statement":
        return "PAID_FULL / PAID_PARTIAL / PARTIAL / UNPAID / PENDING / NO_BALANCE / CREDIT_BALANCE / CARRY_FORWARD."
    if col == "payment_status" and table == "fact_payment":
        return "POSTED / RETURNED / PENDING."
    if col == "amount_usd" and table == "fact_payment_allocation":
        return "Amount of this payment applied to this statement (always positive)."
    if col == "unapplied_amount":
        return "Payment amount not applied to any statement (account credit)."
    if col == "paid_amount" and table == "fact_statement":
        return "Amount paid against this statement (payment waterfall, oldest due first)."
    if col == "paid_amount" and table == "fact_installment_schedule":
        return "Amount paid against this instalment."
    if col == "billed_amount":
        return "Amount newly billed this cycle (charges + interest + fees)."
    if col == "last_payment_date":
        return "Date of the last payment allocated to this statement."
    if col == "replaced_date":
        return "Date the card was replaced (last valid use date)."
    if col in COMMON:
        return COMMON[col]
    if col.endswith("_id"):
        return f"Identifier for {col[:-3].replace('_', ' ')}."
    if col.endswith("_pct"):
        return "Percentage / ratio value."
    if col.endswith("_amount") or col.endswith("_usd") or col.endswith("_limit"):
        return "Monetary amount in USD."
    if col.startswith("hashed_"):
        return "SHA-256 pseudonymised identifier (64 hex chars)."
    if col.startswith("is_") or col.startswith("has_"):
        return "Boolean flag."
    return ""


def table_shape(df) -> tuple[int, int]:
    """(rows, columns) for a DataFrame or a LazyFrame (row count via a lazy scan)."""
    if isinstance(df, pl.LazyFrame):
        return (int(df.select(pl.len()).collect().item()), len(df.collect_schema().names()))
    return len(df), len(df.columns)


def column_stats(df):
    """Yield (column, dtype, null_count, row_count, distinct) for each column.

    A LazyFrame is profiled one column at a time so the table is never
    materialised; dtypes are reported as the pandas dtypes a full load gives.
    """
    if isinstance(df, pl.LazyFrame):
        dtypes = df.head(0).collect().to_pandas().dtypes
        rows = int(df.select(pl.len()).collect().item())
        for c in df.collect_schema().names():
            nulls, nuniq = df.select(
                pl.col(c).null_count().alias("n"), pl.col(c).n_unique().alias("u")
            ).collect(engine="streaming").row(0)
            yield c, str(dtypes[c]), int(nulls), rows, int(nuniq) - (1 if nulls else 0)
        return
    for c in df.columns:
        yield (c, str(df[c].dtype), int(df[c].isna().sum()), len(df),
               int(df[c].nunique(dropna=True)))


def build(tables: dict[str, pd.DataFrame | pl.LazyFrame]) -> tuple[pd.DataFrame, str]:
    rows = []
    shapes = {}
    for name, df in tables.items():
        shapes[name] = table_shape(df)
        for c, dtype, nulls, n, distinct in column_stats(df):
            null_frac = nulls / n if n else float("nan")
            rows.append(dict(
                table=name,
                column=c,
                dtype=dtype,
                nullable=bool(nulls),
                null_pct=round(float(null_frac * 100), 2),
                distinct=distinct,
                description=_describe(name, c),
            ))
    dd = pd.DataFrame(rows)

    lines = ["# Data dictionary", "",
             "Synthetic US credit-card book. All monetary fields are USD; all dates are "
             f"within {C.PERIOD_START} to {C.PERIOD_END} unless stated otherwise.", ""]

    layer_of = {}
    for t in tables:
        if t.startswith("dim_"):
            layer_of[t] = "Dimension"
        elif t.startswith("fact_"):
            layer_of[t] = "Fact"
        else:
            layer_of[t] = "Raw operational mirror"

    for layer in ("Dimension", "Fact", "Raw operational mirror"):
        subset = [t for t in tables if layer_of[t] == layer]
        if not subset:
            continue
        lines += [f"## {layer} tables", ""]
        for t in subset:
            n_rows, n_cols = shapes[t]
            lines += [f"### `{t}`", "",
                      f"- Rows: **{n_rows:,}**",
                      f"- Columns: **{n_cols}**"]
            if t in FACT_NOTES:
                lines.append(f"- Grain: {FACT_NOTES[t].split(': ', 1)[-1]}")
            lines += ["", "| Column | Type | Null % | Distinct | Description |", "|---|---|---|---|---|"]
            sub = dd[dd.table == t]
            for _, r in sub.iterrows():
                lines.append(f"| `{r.column}` | {r.dtype} | {r.null_pct}% | {r.distinct:,} | {r.description} |")
            lines.append("")
    return dd, "\n".join(lines) + "\n"
