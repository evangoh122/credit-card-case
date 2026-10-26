"""Raw operational mirror.

Rebuilds the *shape* of the original Tyme BNPL source tables as a US credit-card
issuer would store them: same table names/columns (with `bnpl_` replaced by
`card_`), DynamoDB-style attribute wrappers on the legacy tables, nested JSON
payloads on the API/BFF tables, and US merchants, cities, banks and USD.

Everything is derived from the star schema so the two layers always reconcile.
"""
from __future__ import annotations

import datetime as dt
import uuid

import numpy as np
import pandas as pd

from . import config as C
from .utils import hash_id, json_dumps, rng_for

def det_uuid(rng):
    """Deterministic UUID so raw output is reproducible."""
    return uuid.UUID(int=int(rng.integers(0, 1 << 63)))


TENANT_HASH = hash_id("tenant:US_CARD_ISSUER")
PROVIDER_HASH = hash_id("provider:US_CARD_ISSUER")


# --------------------------------------------------------------------------
# DynamoDB-style helpers
# --------------------------------------------------------------------------
def S(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "{}"
    return json_dumps({"S": str(v)})


def N(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "{}"
    return json_dumps({"N": str(v)})


def BOOL(v):
    return json_dumps({"BOOL": bool(v)})


def _ts(d) -> str:
    if pd.isna(d):
        return None
    t = pd.Timestamp(d)
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def _micro(d) -> str:
    t = pd.Timestamp(d)
    return str(int(t.value // 1000))


def _acct_hash(acc: str) -> str:
    return hash_id(f"account:{acc}")


def _profile_hash(cust: str) -> str:
    return hash_id(f"profile:{cust}")


def _user_hash(cust: str) -> str:
    return hash_id(f"user:{cust}")


# --------------------------------------------------------------------------
# card_onboarding  (was bnpl_onboarding)
# --------------------------------------------------------------------------
def build_onboarding(cust: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for i, r in cust.reset_index(drop=True).iterrows():
        created = pd.Timestamp(r.onboarding_date) + pd.Timedelta(
            hours=int(rng.integers(6, 22)), minutes=int(rng.integers(0, 60)))
        status = str(rng.choice(
            ["ACCOUNT_CREATED", "INCOME_COLLECTED", "INCOME_SKIPPED", "INIT", "CARD_DECLINED"],
            p=[0.58, 0.16, 0.13, 0.08, 0.05]))
        partner = {
            "loanApplicationId": hash_id(f"app:{r.customer_id}"),
            "requestId": str(det_uuid(rng)),
            "identity": {
                "fullName": hash_id(f"name:{r.customer_id}"),
                "firstName": hash_id(f"first:{r.customer_id}"),
                "middleName": hash_id(f"middle:{r.customer_id}"),
                "lastName": hash_id(f"last:{r.customer_id}"),
                "dateOfBirth": str(r.date_of_birth),
                "ssn": hash_id(f"ssn:{r.customer_id}"),
            },
            "contact": {
                "email": hash_id(f"email:{r.customer_id}"),
                "mobile": hash_id(f"phone:{r.customer_id}"),
                "address": hash_id(f"addr:{r.customer_id}"),
            },
            "employment": {
                "employerName": hash_id(f"emp:{r.employer or 'NA'}"),
                "occupation": r.occupation,
                "annualIncome": float(r.income_annual),
            },
            "decision": {"creditScore": int(r.credit_score), "outcome": status},
        }
        rows.append(dict(
            created_date=S(_ts(created)),
            last_updated_date=S(_ts(created + pd.Timedelta(minutes=int(rng.integers(2, 55))))),
            data_partner=S(json_dumps(partner)) if rng.random() > 0.12 else "{}",
            dob=S(str(r.date_of_birth)),
            gender=N(1 if r.gender == "M" else 0) if rng.random() > 0.35 else "{}",
            has_gender=BOOL(True),
            has_ssn=BOOL(rng.random() < 0.72),
            id_type=S(r.id_type),
            income=N(int(r.income_annual)),
            email_verified="{}",
            is_consent=BOOL(True),
            is_email_verified=BOOL(bool(r.is_email_verified)),
            is_tnc=BOOL(True),
            match_dob=BOOL(True) if rng.random() > 0.08 else "{}",
            match_id_number=BOOL(True) if rng.random() > 0.10 else "{}",
            match_full_name=BOOL(True) if rng.random() > 0.08 else "{}",
            ratio_expect=N(80),
            ratio_fullname=N(100 if rng.random() > 0.12 else 92),
            skip_income=BOOL(status == "INCOME_SKIPPED") if status == "INCOME_SKIPPED" else "{}",
            status=S(status),
            tnc_version=S("3"),
            decline_income=BOOL(True) if status == "CARD_DECLINED" else "{}",
            Operation="INSERT" if rng.random() < 0.5 else "UPDATE",
            WriteTimestampMicros=_micro(created),
            WriteTimestamp=_ts(created).replace("T", " "),
            IngestTimestamp="2026-10-01 06:40:12.115",
            hashed_current_account_no=S(hash_id(f"dda:{r.customer_id}")),
            hashed_email=S(hash_id(f"email:{r.customer_id}")),
            hashed_full_name=S(hash_id(f"name:{r.customer_id}")),
            hashed_id_number=S(hash_id(f"gov_id:{r.customer_id}")),
            hashed_loan_application_id=S(hash_id(f"app:{r.customer_id}")),
            hashed_card_account=S(hash_id(f"cardacct:{r.customer_id}")),
            hashed_profile_id=S(_profile_hash(r.customer_id)),
            hashed_reference_number=S(hash_id(f"ref:{r.customer_id}")),
            hashed_ssn_number=S(hash_id(f"ssn:{r.customer_id}")) if rng.random() > 0.28 else "{}",
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# card_whitelist  (credit limit offers)
# --------------------------------------------------------------------------
def build_whitelist(acct: pd.DataFrame, cust: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    cidx = cust.set_index("customer_id")
    for _, a in acct.iterrows():
        c = cidx.loc[a.customer_id]
        created = pd.Timestamp(a.open_date) - pd.Timedelta(days=int(rng.integers(0, 3)))
        rows.append(dict(
            created_date=S(_ts(created)),
            created_offer_time=S(created.strftime("%Y-%m-%dT%H:%M:%SZ")),
            facility=S(str(round(float(a.initial_credit_limit), 1))),
            offer_status=S("ACTIVE"),
            product_code=S(a.product_id),
            status=S("ACTIVE" if a.account_status != "CLOSED" else "INACTIVE"),
            updated_date=S(_ts(created + pd.Timedelta(seconds=int(rng.integers(30, 900))))),
            updated_offer_time=S(created.strftime("%Y-%m-%dT%H:%M:%SZ")),
            Operation="INSERT",
            WriteTimestampMicros=_micro(created),
            WriteTimestamp=_ts(created).replace("T", " "),
            IngestTimestamp="2026-10-01 06:42:03.552",
            hashed_offer_id=S(hash_id(f"offer:{a.account_id}")),
            hashed_profile_id=S(_profile_hash(a.customer_id)),
            hashed_tenant=S(TENANT_HASH),
            hashed_risk_segment=S(hash_id(f"risk:{c.risk_tier}")),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# card_account
# --------------------------------------------------------------------------
def build_account(acct: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for _, a in acct.iterrows():
        rows.append(dict(
            ID=a.account_id,
            ACCOUNT_STATUS=a.account_status,
            card_PRODUCT_ID=a.product_id,
            ACCEPTED_TERM_AND_CONDITION="true",
            IN_ARREAR=str(int(a.days_past_due > 0)).lower(),
            FIRST_PAYMENT="1" if a.lifetime_payments > 0 else "{}",
            AMOUNT_PURCHASE_TO_DATE=f"{a.lifetime_purchases:.2f}",
            CREATED_DATE=str(pd.Timestamp(a.open_date).date()),
            LAST_UPDATED_DATE=str(pd.Timestamp(a.last_statement_date).date())
            if pd.notna(a.last_statement_date) else str(pd.Timestamp(a.open_date).date()),
            VERSION=str(int(rng.integers(1, 8))),
            NEXT_PAYMENT_DATE=str((pd.Timestamp(a.last_statement_date) + pd.Timedelta(days=25)).date())
            if pd.notna(a.last_statement_date) else "{}",
            OUTSTANDING_BALANCE=f"{a.current_balance:.2f}",
            card_LIMIT=f"{a.credit_limit:.2f}",
            NEXT_INSTALLMENT_NUMBER="0",
            CURRENT_TERM="0",
            hashed_ACCOUNT_NUMBER=_acct_hash(a.account_id),
            hashed_PROVIDER_NAME=PROVIDER_HASH,
            hashed_TENANT_ID=TENANT_HASH,
            hashed_ACCOUNT_HOLDER_KEY=hash_id(f"holder:{a.customer_id}"),
            hashed_LOAN_APPLICATION_ID=hash_id(f"app:{a.customer_id}"),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# card_activation
# --------------------------------------------------------------------------
def build_activation(acct: pd.DataFrame, rng) -> pd.DataFrame:
    sample = acct.sample(frac=min(1.0, 0.82), random_state=C.SEED + 7)
    rows = []
    for _, a in sample.iterrows():
        ts = pd.Timestamp(a.open_date) + pd.Timedelta(hours=int(rng.integers(1, 72)))
        rows.append(dict(
            creation_date=S(_ts(ts)),
            last_modified_date=S(_ts(ts)),
            card_limit=S(str(round(float(a.credit_limit), 1))),
            status=S("ACCOUNT_CREATED" if a.account_status != "CLOSED" else "CLOSED"),
            Operation="INSERT",
            WriteTimestampMicros=_micro(ts),
            WriteTimestamp=_ts(ts).replace("T", " "),
            IngestTimestamp="2026-10-01 06:38:57.204",
            hashed_profile_id=S(_profile_hash(a.customer_id)),
            hashed_deposit_account_number=S(hash_id(f"dda:{a.customer_id}")),
            hashed_card_account_number=S(_acct_hash(a.account_id)),
            hashed_tenant_id=S(TENANT_HASH),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# card_purchase  (authorisations)
# --------------------------------------------------------------------------
def build_purchase(tx: pd.DataFrame, acct: pd.DataFrame, merch: pd.DataFrame,
                   rng, card_last4: dict | None = None) -> pd.DataFrame:
    m = merch.set_index("merchant_id")
    acct_prod = acct.set_index("account_id")["product_id"].to_dict()
    rows = []
    for _, t in tx.iterrows():
        av = float(t.amount_usd)
        md = m.loc[t.merchant_id]
        meta = {
            "amountToPay": f"{av:.2f}",
            "country": md.country_code,
            "purpose": "Payment",
            "referenceLabel": str(int(rng.integers(100000, 999999))),
            "city": str(md.city),
            "state": str(md.state_code or ""),
            "postalCode": str(md.zip_code or ""),
            "accountName": str(md.merchant_name),
            "bankBic": str(rng.choice(C.US_BANKS)[1]),
            "mcc": str(t.mcc),
            "CURRENCY_CODE": t.currency_code,
            "QR_MERCHANT_NAME": str(md.merchant_name),
            "merchantName": str(md.merchant_name),
            "paymentType": "P2M",
            "creditType": "QP2M" if md.country_code == "US" else "CROSSBORDER",
            "TERM": "0",
            "SHORT_CODE": "QUS",
            "channel": str(t.channel_group),
            "cardLast4": str((card_last4 or {}).get(t.card_id, "")),
            "transactionType": t.transaction_type,
        }
        if t.transaction_type == "REFUND":
            status = "REFUND_CONFIRMED"
        else:
            status = {"APPROVED": "PAYMENT_CONFIRMED",
                      "DECLINED": "PAYMENT_DECLINED",
                      "REVERSED": "PAYMENT_REVERSED"}.get(t.status, "PENDING")
        rows.append(dict(
            ID=t.transaction_id,
            MERCHANT_ID=t.merchant_id,
            STORE_ID="{}",
            PRODUCT_TYPE=acct_prod.get(t.account_id, "CREDIT_CARD"),
            AMOUNT=f"{av:.2f}",
            BILLING_AMOUNT=f"{av:.2f}",
            SETTLEMENT_AMOUNT=f"{av:.2f}" if t.status == "APPROVED" else "0.00",
            ORIGINAL_REFERENCE=str(rng.integers(10**12, 10**13)),
            STATUS=status,
            CREATED_DATE=str(pd.Timestamp(t.transaction_ts).replace(microsecond=0)),
            LAST_UPDATED_DATE=str(pd.Timestamp(t.transaction_ts).replace(microsecond=0)
                                   + pd.Timedelta(seconds=int(rng.integers(0, 8)))),
            METADATA=json_dumps(meta),
            OPERATION_TYPE="{}",
            ORIGINAL_PURCHASE_ID="{}",
            NOTE="{}",
            VERSION="1",
            hashed_TRANSACTION_IDENTIFIER=hash_id(f"txn:{t.transaction_id}"),
            hashed_ORIGINAL_ID=hash_id(f"orig:{t.transaction_id}"),
            hashed_ACCOUNT_NUMBER=_acct_hash(t.account_id),
            hashed_PROVIDER_NAME=PROVIDER_HASH,
            hashed_TENANT_ID=TENANT_HASH,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# card_voucher_qr_code  (merchant / location payload)
# --------------------------------------------------------------------------
def build_qr_code(tx: pd.DataFrame, cust: pd.DataFrame, merch: pd.DataFrame, rng) -> pd.DataFrame:
    m = merch.set_index("merchant_id")
    c = cust.set_index("customer_id")
    qr = tx[tx.channel_group.isin(["IN_STORE", "ONLINE", "MOBILE_APP"])].copy()
    if len(qr) > C.RAW_TXN_CAP:
        qr = qr.sample(C.RAW_TXN_CAP, random_state=C.SEED + 11)
    rows = []
    for _, t in qr.iterrows():
        md = m.loc[t.merchant_id]
        cu = c.loc[t.customer_id]
        bank = str(rng.choice(C.US_BANKS)[1])
        failed = t.status == "DECLINED"
        rows.append(dict(
            ID=t.transaction_id,
            STATUS=("REFUND_COMPLETED" if t.transaction_type == "REFUND"
                    else ("COMPLETED" if t.status == "APPROVED" else ("FAILED" if failed else "REVERSED"))),
            AMOUNT=f"{float(t.amount_usd):.2f}",
            CURRENCY_CODE=t.currency_code,
            MERCHANT_NAME=str(md.merchant_name),
            TRANSACTION_TIME=str(pd.Timestamp(t.transaction_ts).replace(microsecond=0)),
            card_OPTION_TERM="0",
            OPTION_ENCODED_KEY=str(det_uuid(rng)),
            EXPIRY_DATE=str(pd.Timestamp(t.transaction_ts) + pd.Timedelta(minutes=15)),
            FAILURE_REASON="Issuer decline" if failed else None,
            FAILURE_REASON_CODE="051" if failed else None,
            CREATED_DATE=str(pd.Timestamp(t.transaction_ts) - pd.Timedelta(seconds=2)),
            LAST_UPDATED_DATE=str(pd.Timestamp(t.transaction_ts) + pd.Timedelta(seconds=3)),
            VERSION="2",
            COUNTRY=md.country_code,
            COUNTRY_NAME=md.country_code,
            LOCAL_INSTRUMENT=None,
            CITY=str(md.city),
            BANK_BIC=bank,
            PROXY_NOTIFY_FLAGS=str(rng.choice(["010", "110", "310"])),
            MCC=str(t.mcc),
            CREDIT_TYPE="QP2M",
            CATEGORY_CODE=str(t.mcc),
            QR_PAYMENT_TYPE="P2M",
            ROUTING_CODE=bank,
            CITY_OF_BIRTH=str(cu.city),
            COUNTRY_OF_BIRTH="US",
            BIRTH_DAY=str(cu.date_of_birth),
            hashed_MERCHANT_BANK_ACCOUNT=hash_id(f"mba:{t.merchant_id}"),
            hashed_VOUCHER_NUMBER=hash_id(f"vch:{t.transaction_id}"),
            hashed_IDEMPOTENCY_KEY=hash_id(f"idem:{t.transaction_id}"),
            hashed_REFERENCE_ID=hash_id(f"ref:{t.transaction_id}"),
            hashed_ACQUIRER_REQUIRED_INFO=hash_id(f"acq:{t.merchant_id}") if rng.random() > 0.35 else None,
            hashed_DEPOSIT_ACCOUNT_NUMBER=_acct_hash(t.account_id),
            hashed_TRANSACTION_REFERENCE=hash_id(f"tr:{t.transaction_id}"),
            hashed_PROFILE_ID=_profile_hash(t.customer_id),
            hashed_NAME=hash_id(f"name:{t.customer_id}"),
            hashed_MERCHANT_ID=hash_id(f"merch:{t.merchant_id}"),
            hashed_INSTR_ID=hash_id(f"instr:{t.transaction_id}"),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# card_voucher / card_voucher_activity
# --------------------------------------------------------------------------
def build_voucher(tx: pd.DataFrame, cust: pd.DataFrame, merch: pd.DataFrame, rng) -> pd.DataFrame:
    m = merch.set_index("merchant_id")
    c = cust.set_index("customer_id")
    v = tx[tx.transaction_type == "PURCHASE"].copy()
    if len(v) > C.RAW_TXN_CAP:
        v = v.sample(C.RAW_TXN_CAP, random_state=C.SEED + 13)
    rows = []
    for _, t in v.iterrows():
        md = m.loc[t.merchant_id]
        amt = abs(float(t.amount_usd))
        meta = {
            "amountToPay": f"{amt:.2f}", "country": md.country_code, "purpose": "Payment",
            "city": str(md.city), "state": str(md.state_code or ""),
            "accountName": str(md.merchant_name), "mcc": str(t.mcc),
            "CURRENCY_CODE": t.currency_code, "QR_MERCHANT_NAME": str(md.merchant_name),
            "merchantName": str(md.merchant_name), "paymentType": "P2M", "TERM": "0",
        }
        rows.append(dict(
            ID=f"VCH-{t.transaction_id}",
            PRODUCT_TYPE="CREDIT_CARD",
            SPENT_AMOUNT=f"{amt:.2f}",
            EXPIRY_DATE=str(pd.Timestamp(t.transaction_ts) + pd.Timedelta(minutes=15)),
            STATUS="REDEEM_CONFIRMED" if t.status == "APPROVED" else "REDEEM_ERROR",
            CREATED_DATE=str(pd.Timestamp(t.transaction_ts) - pd.Timedelta(seconds=6)),
            LAST_UPDATED_DATE=str(pd.Timestamp(t.transaction_ts)),
            MERCHANT_ID=t.merchant_id,
            MERCHANT_NAME=str(md.merchant_name),
            PURCHASE_ID=t.transaction_id,
            PURCHASE_DATE=str(pd.Timestamp(t.transaction_ts)),
            STORE_ID=None,
            TERMINAL_ID=None,
            REQUEST_ID=str(det_uuid(rng)),
            VERSION="5",
            NOTE=None,
            ACTIVITY_TYPE="CONFIRM",
            LOCATION=f"{md.city}, {md.state_code}" if md.state_code else str(md.city),
            CHANNEL="E_COMMERCE" if t.channel_group == "ONLINE" else "POS",
            PAID_AMOUNT=f"{amt:.2f}" if t.status == "APPROVED" else "0.00",
            REMAINING_AMOUNT="0.00",
            TRANSACTION_IDENTIFIER=f"CC{int(rng.integers(10**14, 10**15))}",
            FINALLIZED_TRANSACTION_IDENTIFIER=None,
            ADDITIONAL_INFO=json_dumps(meta),
            TIME=str(pd.Timestamp(t.transaction_ts)),
            APPROVED_REFUND_AMOUNT=None,
            REQUEST_REFUND_AMOUNT=None,
            REFUND_TRANSACTION_IDENTIFIER=None,
            REFUND_FINALLIZED_TRANSACTION_IDENTIFIER=None,
            TOTAL_APPROVED_REFUND_AMOUNT=None,
            OPERATION_TYPE="REDEEM",
            TENANT_ID="usCards",
            hashed_VOUCHER_NUMBER=hash_id(f"vch:{t.transaction_id}"),
            hashed_card_ACCOUNT_NUMBER=_acct_hash(t.account_id),
            hashed_ORIGINAL_ID=hash_id(f"orig:{t.transaction_id}"),
        ))
    return pd.DataFrame(rows)


def build_voucher_activity(vouchers: pd.DataFrame, rng) -> pd.DataFrame:
    v = vouchers.copy()
    rows = []
    n = 0
    for _, t in v.iterrows():
        base = pd.Timestamp(t.PURCHASE_DATE)
        for k, (act, st, delta) in enumerate([
            ("GENERATE", "GENERATED", -6),
            ("REDEEM", "PREPARE_REDEEM", -3),
            ("REDEEM", "REDEEMED" if t.STATUS == "REDEEM_CONFIRMED" else "REDEEM_ERROR", 0),
        ]):
            n += 1
            rows.append(dict(
                ID=f"VAC{n:09d}",
                VOUCHER_ID=t.ID,
                REQUEST_ID=str(det_uuid(rng)),
                TRANSACTION_IDENTIFIER="{}".format(t.TRANSACTION_IDENTIFIER),
                ACTIVITY_TYPE=act,
                STATUS=st,
                NOTE=None,
                SPENT_AMOUNT=("{:.2f}".format(float(t.PAID_AMOUNT))) if k == 2 else None,
                CREATED_DATE=str(base + pd.Timedelta(seconds=delta)),
                LAST_UPDATED_DATE=str(base + pd.Timedelta(seconds=delta)),
                REQUEST_REFUND_AMOUNT=None,
                APPROVED_REFUND_AMOUNT=None,
                TOTAL_APPROVED_REFUND_AMOUNT=None,
                FINALLIZED_TRANSACTION_IDENTIFIER=None,
                REFUND_TRANSACTION_IDENTIFIER=None,
                REFUND_FINALLIZED_TRANSACTION_IDENTIFIER=None,
                OPERATION_TYPE=None,
                hashed_VOUCHER_NUMBER=t.hashed_VOUCHER_NUMBER,
                hashed_ORIGINAL_ID=t.hashed_ORIGINAL_ID,
                hashed_TENANT_ID=TENANT_HASH,
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pay-over-time: card_payment_option / card_purchase_schedule /
# card_purchase_activity / card_loan_account
# --------------------------------------------------------------------------
def build_payment_option(plan: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for _, p in plan.iterrows():
        term = int(p.term_months)
        monthly = float(p.monthly_payment)
        inst = [{"installmentOrder": k, "purchaseAmount": round(monthly, 2),
                 "dueDate": int((pd.Timestamp(p.plan_start_date) + pd.Timedelta(days=int(k * 30.44))).value // 10**6)}
                for k in range(1, term + 1)]
        opt = {"encodedKey": str(det_uuid(rng)), "totalPayment": float(p.total_repayment),
               "interestPercent": float(p.promo_apr), "interestAmount": float(p.total_interest_fee),
               "term": term, "installments": inst}
        rows.append(dict(
            ID=p.plan_id,
            ENCODED_KEY=opt["encodedKey"],
            AMOUNT=f"{float(p.principal_amount):.2f}",
            TERM=str(term),
            OPTION_DATA=json_dumps(opt),
            MERCHANT_DATA="{}",
            CREATED_DATE=str(pd.Timestamp(p.plan_start_date)),
            LAST_UPDATED_DATE=str(pd.Timestamp(p.plan_start_date)),
            RISK_SEGMENT=json_dumps({"FEE_PERCENT": f"{float(p.promo_apr) * 100:.2f}",
                                     "FEE_ID": hash_id(f"fee:{term}")[:32]}),
            OFFER_DETAIL=None,
            hashed_REFERENCE_ID=hash_id(f"ref:{p.transaction_id}"),
            hashed_ACCOUNT_NUMBER=_acct_hash(p.account_id),
            hashed_TENANT_ID=TENANT_HASH,
        ))
    return pd.DataFrame(rows)


def build_purchase_schedule(sched: pd.DataFrame, plan: pd.DataFrame, rng) -> pd.DataFrame:
    pm = plan.set_index("plan_id")
    rows = []
    for _, s in sched.iterrows():
        p = pm.loc[s.plan_id]
        rows.append(dict(
            ID=s.schedule_id,
            PURCHASE_ID=p.transaction_id,
            REQUEST_TRANSACTION_ID=str(det_uuid(rng)) if s.status != "PENDING" else None,
            RESPONSE_TRANSACTION_ID=str(det_uuid(rng)) if s.status != "PENDING" else None,
            AMOUNT=f"{float(s.amount_due):.2f}",
            TYPE=f"INSTALLMENT_{int(s.installment_number)}",
            DUE_DATE=str(pd.Timestamp(s.due_date)),
            STATUS=s.status,
            MESSAGE=None,
            ENTRY_DATE=str(pd.Timestamp(s.paid_date)) if pd.notna(s.paid_date) else None,
            CREATED_DATE=str(pd.Timestamp(p.plan_start_date)),
            LAST_UPDATED_DATE=str(pd.Timestamp(s.paid_date) if pd.notna(s.paid_date)
                                  else pd.Timestamp(p.plan_start_date)),
            METADATA=None,
            PAID_AMOUNT=f"{float(s.paid_amount):.2f}",
            FEE_EXPECTED=f"{float(s.interest_fee_component):.2f}",
            INTEREST_EXPECTED="0.00",
            PRINCIPAL_AMOUNT=f"{float(s.principal_component):.2f}",
            UPFRONT_FEE_AMOUNT="0.00",
            LATE_FEE_AMOUNT="0.00" if s.status != "LATE" else "29.00",
            PENALTY_AMOUNT="0.00",
            hashed_ACCOUNT_NUMBER=_acct_hash(s.account_id),
            hashed_TENANT_ID=TENANT_HASH,
        ))
    return pd.DataFrame(rows)


def build_purchase_activity(sched: pd.DataFrame, rng, plan: pd.DataFrame | None = None) -> pd.DataFrame:
    txn_of = {}
    if plan is not None:
        txn_of = plan.set_index("plan_id")["transaction_id"].to_dict()
    rows = []
    n = 0
    for _, s in sched.iterrows():
        base = pd.Timestamp(s.due_date) - pd.Timedelta(days=25)
        for act_status in (["PENDING", "PAID"] if s.status == "PAID" else [s.status]):
            n += 1
            rows.append(dict(
                ID=f"PAC{n:09d}",
                PURCHASE_SCHEDULE_ID=s.schedule_id,
                PURCHASE_ID=txn_of.get(s.plan_id, s.plan_id),
                ORIGINAL_REFERENCE=str(rng.integers(10**12, 10**13)),
                TYPE=None,
                PURCHASE_SCHEDULE_STATUS=act_status,
                PURCHASE_SCHEDULE_TYPE=f"INSTALLMENT_{int(s.installment_number)}",
                AMOUNT=f"{float(s.amount_due):.2f}",
                REASON=None,
                MERCHANT_ID=None,
                CREATED_DATE=str(base),
                LAST_UPDATED_DATE=str(pd.Timestamp(s.paid_date) if pd.notna(s.paid_date) else base),
                PAID_AMOUNT=f"{float(s.paid_amount):.2f}" if act_status == "PAID" else "0.00",
                hashed_REFERENCE=hash_id(f"ref:{s.schedule_id}"),
                hashed_TENANT_ID=TENANT_HASH,
            ))
    return pd.DataFrame(rows)


def build_loan_account(sched: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    pm = plan.set_index("plan_id", drop=False)
    rows = []
    for _, s in sched.iterrows():
        p = pm.loc[s.plan_id]
        paid = float(s.paid_amount)
        due = float(s.amount_due)
        late = s.status == "LATE"
        dpd = 0 if not late else 35
        rows.append(dict(
            pk=S(f"PLAN#{p.plan_id}"),
            sk=S(f"INSTALLMENT#{int(s.installment_number)}"),
            createdAt=S(pd.Timestamp(p.plan_start_date).strftime("%Y-%m-%dT%H:%M:%SZ")),
            dueDate=S(pd.Timestamp(s.due_date).strftime("%Y-%m-%dT%H:%M:%S+00:00")),
            entryDate=S(pd.Timestamp(s.paid_date).strftime("%Y-%m-%dT%H:%M:%S+00:00"))
            if pd.notna(s.paid_date) else "{}",
            feeDue=S("0.00"),
            feeExpected=S(f"{float(s.interest_fee_component):.2f}"),
            feePaid=S(f"{float(s.interest_fee_component):.2f}" if paid > 0 else "0.00"),
            feesBalance=S("0.00"),
            feesDue=S("0.00"),
            feesPaid=S(f"{float(s.interest_fee_component):.2f}" if paid > 0 else "0.00"),
            icDaysInArrears=N(dpd),
            interestDue=S("0.00"),
            interestExpected=S("0.00"),
            interestPaid=S("0.00"),
            lateFeeAmount=S("0.00" if not late else "29.00"),
            loanAmount=S(f"{float(p.principal_amount):.2f}"),
            number=S(str(int(s.installment_number))),
            paidLateFee="{}",
            paidPenaltyFee=S("0.00"),
            penaltyDue=S("0.00" if not late else "29.00"),
            penaltyExpected=S("0.00"),
            penaltyFee=S("0.00" if not late else "29.00"),
            penaltyPaid=S("0.00"),
            principalAmount=S(f"{float(s.principal_component):.2f}"),
            principalBalance=S(f"{max(0.0, float(p.principal_amount) - float(s.principal_component) * int(s.installment_number)):.2f}"),
            principalDue=S(f"{float(s.principal_component):.2f}" if paid <= 0 else "0.00"),
            principalExpected=S(f"{float(s.principal_component):.2f}"),
            principalPaid=S(f"{float(s.principal_component):.2f}" if paid > 0 else "0.00"),
            productCode=S(p.product_name if hasattr(p, "product_name") else "PAY_OVER_TIME"),
            purchaseId=S(p.transaction_id),
            referenceKey=S(hash_id(f"ref:{s.schedule_id}")[:32]),
            state=S({"PAID": "PAID", "PENDING": "PENDING", "LATE": "LATE",
                     "PARTIAL": "PARTIALLY_PAID"}.get(s.status, "PENDING")),
            status=S("ACTIVE" if p.plan_status == "ACTIVE" else "CLOSED"),
            totalBalance=S(f"{float(p.remaining_balance):.2f}"),
            triggeredAging=BOOL(late),
            updatedAt=S(pd.Timestamp(s.due_date).strftime("%Y-%m-%dT%H:%M:%SZ")),
            upfrontFeeAmount=S("0.00"),
            daysInArrears=S(str(dpd)),
            daysLate=S("0" if not late else "5"),
            totalDue=S(f"{due - paid:.2f}"),
            lastSetToArrearsDate=S(pd.Timestamp(s.due_date).strftime("%Y-%m-%dT%H:%M:%SZ")) if late else "{}",
            accountSubState="{}",
            deferredFeeIncome="{}",
            closedDate="{}",
            Operation="UPDATE",
            WriteTimestampMicros=str(int(pd.Timestamp(s.due_date).value // 1000)),
            WriteTimestamp=pd.Timestamp(s.due_date).strftime("%Y-%m-%d %H:%M:%S"),
            IngestTimestamp="2026-10-01 06:55:19.402",
            hashed_cardAccount=S(_acct_hash(s.account_id)),
            hashed_productPlatform=S(hash_id("platform:pay_over_time")),
            hashed_tenant=S(TENANT_HASH),
            hashed_loanAccountNumber=S(hash_id(f"loanacct:{s.plan_id}")),
            hashed_loanApplicationId=S(hash_id(f"app:{s.customer_id}")),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# card_payment_workflow
# --------------------------------------------------------------------------
def build_payment_workflow(pay: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    steps = ["AUTHORIZE", "TRANSFER", "FINALIZE"]
    for _, p in pay.iterrows():
        amt = float(p.amount_usd)
        for k, action in enumerate(steps):
            status = {"AUTHORIZE": "AUTHORIZED", "TRANSFER": "SETTLEMENTED", "FINALIZE": "POST_SUCCEED"}[action]
            ts = pd.Timestamp(p.payment_date) + pd.Timedelta(minutes=k * 2 + int(rng.integers(0, 3)))
            req = {
                "paymentReferenceId": hash_id(f"payref:{p.payment_id}"),
                "paymentOrderId": hash_id(f"order:{p.payment_id}-{k}")[:32] if k > 0 else None,
                "requestedDomain": "card",
                "tenantId": TENANT_HASH,
                "source": hash_id(f"src:{p.account_id}"),
                "action": action,
                "paymentOrder": {
                    "paymentUseCase": "CardPayment",
                    "paymentDefinition": "CARD_statementRepayment",
                    "timestamp": int(ts.value // 10**6),
                    "amount": amt,
                    "currency": "USD",
                    "referenceNumber": p.reference_number,
                },
            }
            resp = {
                "paymentReferenceId": req["paymentReferenceId"],
                "paymentOrderId": hash_id(f"order:{p.payment_id}-{k}")[:32],
                "bankPaymentOrderId": hash_id(f"bank:{p.payment_id}-{k}")[:32],
                "requestedDomain": "card",
                "status": status,
                "processedAt": int(ts.value // 10**6),
            }
            rows.append(dict(
                ID=f"PWF-{p.payment_id}-{k}",
                PURCHASE_ID="{}",
                TASK_TOKEN=str(hash_id(f"task:{p.payment_id}-{k}")),
                PAYMENT_REQUEST=json_dumps(req),
                PAYMENT_RESPONSE=json_dumps(resp),
                PAYMENT_ACTION=action,
                PRODUCT_TYPE="CREDIT_CARD",
                AMOUNT=f"{amt:.3f}",
                STEP={"AUTHORIZE": "ON_HOLD", "TRANSFER": "ON_SETTLEMENT", "FINALIZE": "ON_POST_GL"}[action],
                INSTALLMENT="0",
                STATUS=status,
                FAILURE_REASON_CODE=None,
                FAILURE_REASON_MESSAGE=None,
                VERSION=str(k + 1),
                CREATED_DATE=str(ts),
                LAST_UPDATED_DATE=str(ts + pd.Timedelta(seconds=int(rng.integers(0, 4)))),
                hashed_card_ACCOUNT_NUMBER=_acct_hash(p.account_id),
                hashed_REFERENCE_NUMBER=hash_id(f"ref:{p.payment_id}"),
                hashed_PAYMENT_ORDER_ID=hash_id(f"order:{p.payment_id}-{k}"),
                hashed_BANK_PAYMENT_ORDER_ID=hash_id(f"bank:{p.payment_id}-{k}"),
                hashed_ORIGINAL_ID=hash_id(f"orig:{p.payment_id}-{k}"),
                hashed_TENANT_ID=TENANT_HASH,
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Fraud raw tables
# --------------------------------------------------------------------------
def build_fraud_upload(fraud: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for _, f in fraud.iterrows():
        ts = pd.Timestamp(f.alert_date) + pd.Timedelta(hours=int(rng.integers(0, 12)))
        rows.append(dict(
            created_date=S(_ts(ts)),
            detail=S(""),
            hash_code=S(hash_id(f"fh:{f.alert_id}")),
            ip_address=S(f.ip_address),
            last_updated_date=S(_ts(ts)),
            status=S("SUCCESS"),
            Operation="INSERT",
            WriteTimestampMicros=_micro(ts),
            WriteTimestamp=_ts(ts).replace("T", " "),
            IngestTimestamp="2026-10-01 06:51:44.732",
            hashed_profile_id=S(_profile_hash(f.customer_id)),
            hashed_reference_number=S(hash_id(f"ref:{f.alert_id}")),
            hashed_tenant=S(TENANT_HASH),
        ))
    return pd.DataFrame(rows)


def build_fraud_insights(fraud: pd.DataFrame, cust: pd.DataFrame, rng) -> pd.DataFrame:
    c = cust.set_index("customer_id")
    rows = []
    for _, f in fraud.iterrows():
        cu = c.loc[f.customer_id]
        insights = [
            {"code": "ipInfo", "value": {
                "ip": f.ip_address, "loc": f"{rng.uniform(25, 48):.4f},{rng.uniform(-122, -71):.4f}",
                "city": str(cu.city), "postal": str(cu.zip_code), "region": str(cu.state_code),
                "country": "US", "timezone": str(cu.metro), "privacyTor": "false",
                "privacyVpn": "false", "privacyProxy": "false", "privacyHosting": "false"}},
            {"code": "deviceInfo", "value": {
                "deviceId": hash_id(f"dev:{f.customer_id}")[:16],
                "os": str(rng.choice(["Android 14", "iOS 18.3", "Windows 11"])),
                "isEmulator": "false", "isRooted": "false",
                "screenResolution": str(rng.choice(["1080x2340", "1179x2556", "1920x1080"]))}},
            {"code": "geoVelocity", "value": {
                "distanceKm": int(rng.integers(0, 4200)),
                "minutesSinceLastTxn": int(rng.integers(1, 3000)),
                "impossibleTravel": str(rng.random() < 0.12).lower()}},
            {"code": "velocity", "value": {
                "txnCount1h": int(rng.integers(1, 9)),
                "txnCount24h": int(rng.integers(1, 22)),
                "amount24h": round(float(f.loss_amount) * rng.uniform(1.0, 3.5), 2)}},
            {"code": "contacts", "value": {
                "contactsTotalCount": str(rng.integers(4, 400)),
                "contactsMeetCount": str(rng.integers(0, 3)),
                "phoneNumberHomeCount": str(rng.integers(0, 2))}},
            {"code": "lastApps", "value": [
                {"appPaid": "false", "appPackageName": pkg,
                 "appDateInstalled": (pd.Timestamp(f.alert_date) - pd.Timedelta(days=int(rng.integers(1, 400)))).strftime("%Y-%m-%d %H:%M:%S"),
                 "appGooglePlayCategory": cat, "appDateInstalledRankDesc": "1"}
                for pkg, cat in zip(
                    rng.choice(["com.chase.sig.android", "com.paypal.android.p2pmobile",
                                "com.ubercab", "com.amazon.mShop.android.shopping",
                                "com.venmo", "com.squareup.cash"], 4),
                    rng.choice(["finance", "shopping", "travel", "communication"], 4))]},
        ]
        rows.append(dict(
            created_date=S(_ts(pd.Timestamp(f.alert_date))),
            insights=S(json_dumps(insights)),
            last_updated_date=S(_ts(pd.Timestamp(f.alert_date) + pd.Timedelta(days=2))),
            Operation="UPDATE",
            WriteTimestampMicros=_micro(pd.Timestamp(f.alert_date)),
            WriteTimestamp=_ts(pd.Timestamp(f.alert_date)).replace("T", " "),
            IngestTimestamp="2026-10-01 06:50:11.883",
            hashed_profile_id=S(_profile_hash(f.customer_id)),
            hashed_reference_number=S(hash_id(f"ref:{f.alert_id}")),
            hashed_tenant=S(TENANT_HASH),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# profile_mapping + segment logs
# --------------------------------------------------------------------------
def build_profile_mapping(cust: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(dict(
        hashed_profile_id=[_profile_hash(x) for x in cust.customer_id],
        hashed_user_id=[_user_hash(x) for x in cust.customer_id],
    ))


def build_segment(ev: pd.DataFrame, rng) -> dict[str, pd.DataFrame]:
    out = {}
    for plat, name in [("ANDROID", "segment_logs_android"),
                       ("IOS", "segment_logs_ios"),
                       ("WEB", "segment_logs_backend")]:
        d = ev[ev.platform == plat]
        if d.empty:
            out[name] = pd.DataFrame()
            continue
        rows = []
        for _, e in d.iterrows():
            ts = pd.Timestamp(e.event_timestamp)
            is_client = plat != "WEB"
            ctx = {
                "app": {"build": "2041" if is_client else "0",
                        "name": hash_id("app:us-credit-card"),
                        "namespace": "com.uscard.mobile" if is_client else "backend",
                        "version": str(e.app_version or "n/a")},
            }
            if is_client:
                ctx["device"] = {
                    "id": hash_id(f"device:{e.customer_id}")[:64],
                    "manufacturer": str(e.device).split()[0],
                    "model": str(e.device),
                    "name": hash_id(f"devname:{e.customer_id}")[:64],
                    "type": plat.lower(),
                    "osVersion": e.os_version,
                    "adTrackingEnabled": "true" if plat == "ANDROID" else "false",
                    "advertisingId": hash_id(f"adid:{e.customer_id}")[:36],
                }
                ctx["library"] = {"name": "analytics-android" if plat == "ANDROID" else "analytics-ios",
                                  "version": "1.9.2"}
            else:
                ctx["library"] = {"name": hash_id("lib:backend"), "version": "unknown"}
                ctx["protocols"] = {"sourceId": "USCARD_BE", "omitted": ["id"]}
            props = {
                "flow_conversion": "yes-positive" if e.is_conversion_event else "no",
                "product_area": "cards",
                "product_value_stream": "credit-card",
                "spec_track": "2.1.0",
                "ui_screen_path": e.screen_path,
                "ui_screen_state": "functional",
                "device_id": hash_id(f"device:{e.customer_id}")[:16],
                "product_user_type": "personal",
            }
            rows.append(dict(
                anonymousId=hash_id(f"anon:{e.customer_id}")[:36] if is_client else None,
                channel="client" if is_client else "server",
                context=json_dumps(ctx),
                event=e.event_name.replace("_", " "),
                integrations="{}",
                messageId=hash_id(f"msg:{e.event_id}")[:32] if is_client else hash_id(f"msg:{e.event_id}"),
                originalTimestamp=ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + ("Z" if is_client else "+00:00"),
                projectId="USCARD" + ("_AND" if plat == "ANDROID" else "_IOS" if plat == "IOS" else "_BE"),
                properties=json_dumps(props),
                receivedAt=(ts + pd.Timedelta(milliseconds=int(rng.integers(20, 900)))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                sentAt=(ts + pd.Timedelta(milliseconds=int(rng.integers(1, 200)))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if is_client else None,
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                traits=None,
                type="track",
                version="2",
                writeKey=None,
                _rescued_data=json_dumps({"__segment_internal": {}}),
                hashed_userId=_user_hash(e.customer_id),
                _metadata=json_dumps({"bundled": ["Firebase", "AppsFlyer"], "unbundled": ["AWS S3"]}),
            ))
        out[name] = pd.DataFrame(rows)
    return out


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
def build_all(dims: dict, facts: dict) -> dict[str, pd.DataFrame]:
    rng = rng_for("raw_mirror")
    cust = dims["dim_customer"]
    acct = dims["dim_account"]
    merch = dims["dim_merchant"]
    tx = facts["fact_transaction"]
    plan = facts["fact_installment_plan"]
    sched = facts["fact_installment_schedule"]
    pay = facts["fact_payment"]
    fraud = facts["fact_fraud_alert"]

    out = {}
    # sample the raw operational tables so the mirror stays portable at 50k scale
    tx_raw = tx if len(tx) <= C.RAW_TXN_CAP else tx.sample(C.RAW_TXN_CAP, random_state=C.SEED + 19)
    pay_raw = pay if len(pay) <= C.RAW_PAYMENT_CAP else pay.sample(C.RAW_PAYMENT_CAP, random_state=C.SEED + 21)
    fraud_raw = fraud if len(fraud) <= C.RAW_FRAUD_CAP else fraud.sample(C.RAW_FRAUD_CAP, random_state=C.SEED + 22)

    # keep the raw layer referentially connected: only plans whose originating
    # purchase is inside the sampled transaction extract
    tx_ids = set(tx_raw.transaction_id)
    plan_conn = plan[plan.transaction_id.isin(tx_ids)]
    if len(plan_conn) > C.RAW_PLAN_CAP:
        plan_raw = plan_conn.sample(C.RAW_PLAN_CAP, random_state=C.SEED + 23)
    else:
        plan_raw = plan_conn
    sched_raw = sched[sched.plan_id.isin(set(plan_raw.plan_id))]

    out["card_onboarding"] = build_onboarding(cust, rng)
    out["card_whitelist"] = build_whitelist(acct, cust, rng)
    out["card_account"] = build_account(acct, rng)
    out["card_activation"] = build_activation(acct, rng)
    out["card_purchase"] = build_purchase(
        tx_raw, acct, merch, rng,
        dims["dim_card"].set_index("card_id")["last4"].to_dict())
    out["card_voucher_qr_code"] = build_qr_code(tx_raw, cust, merch, rng)
    out["card_voucher"] = build_voucher(tx_raw, cust, merch, rng)
    out["card_voucher_activity"] = build_voucher_activity(out["card_voucher"], rng)
    out["card_payment_option"] = build_payment_option(plan_raw, rng)
    out["card_purchase_schedule"] = build_purchase_schedule(sched_raw, plan_raw, rng)
    out["card_purchase_activity"] = build_purchase_activity(sched_raw, rng, plan_raw)
    out["card_loan_account"] = build_loan_account(sched_raw, plan_raw)
    out["card_payment_workflow"] = build_payment_workflow(pay_raw, rng)
    if not fraud.empty:
        out["card_fraud_upload"] = build_fraud_upload(fraud_raw, rng)
        out["card_fraud_insights"] = build_fraud_insights(fraud_raw, cust, rng)
    out["profile_mapping"] = build_profile_mapping(cust)
    out.update(build_segment(facts["fact_app_event"], rng))
    return out

