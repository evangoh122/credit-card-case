import datetime as dt
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from build import config as C, dimensions, facts, raw_mirror
from build.facts import (
    Buf,
    _emit_plan_v2,
    _final_dpd,
    _settled_closing_amounts,
)


class SettlementTests(unittest.TestCase):
    def test_card_cash_is_consumed_at_most_once(self):
        # v6 regression, rewritten for the cash-consuming settlement rule:
        # the $150 of card cash settles $100 of the first closing and only the
        # remaining $50 of the second. It must never fund $200 of settlement.
        stmts = [
            {"statement_date": dt.date(2026, 1, 1), "closing_balance": 100.0},
            {"statement_date": dt.date(2026, 2, 1), "closing_balance": 200.0},
            {"statement_date": dt.date(2026, 3, 1), "closing_balance": -10.0},
        ]
        payments = [
            (dt.date(2026, 1, 1), 25.0),   # same-day cash is not later cash
            (dt.date(2026, 1, 15), 50.0),
            (dt.date(2026, 3, 15), 100.0), # later than the next statement cycle
        ]
        settled = _settled_closing_amounts(stmts, payments)
        self.assertEqual(settled, [100.0, 50.0, 0.0])
        self.assertLessEqual(sum(settled), 150.0)

    def test_no_later_cash_leaves_closing_unsettled(self):
        stmts = [{"statement_date": dt.date(2026, 2, 1), "closing_balance": 80.0}]
        self.assertEqual(_settled_closing_amounts(stmts, []), [0.0])


class DelinquencyTests(unittest.TestCase):
    def test_final_payment_cures_account_dpd(self):
        stmt = {
            "due_date_obj": dt.date(2026, 1, 1),
            "min_eff": 35.0,
            "alloc_min": 0.0,
        }
        self.assertEqual(_final_dpd([stmt], dt.date(2026, 1, 31)), 30)
        stmt["alloc_min"] = 35.0
        self.assertEqual(_final_dpd([stmt], dt.date(2026, 1, 31)), 0)

    def test_dpd_uses_supplied_closure_date(self):
        stmt = {
            "due_date_obj": dt.date(2026, 1, 1),
            "min_eff": 35.0,
            "alloc_min": 0.0,
        }
        self.assertEqual(_final_dpd([stmt], dt.date(2026, 1, 20)), 19)


class GeographyTests(unittest.TestCase):
    def test_domestic_merchants_use_only_us_locations(self):
        geo = pd.DataFrame([
            dict(geo_id=1, city="New York", state_code="NY", zip_code="10001",
                 country_code="US", region="Northeast"),
            dict(geo_id=2, city="Toronto", state_code="ON", zip_code="M5V",
                 country_code="CA", region="International"),
        ])
        merchants = dimensions.build_dim_merchant(geo)
        domestic = merchants[~merchants.is_international]
        self.assertTrue((domestic.geo_id == 1).all())
        self.assertTrue((domestic.country_code == "US").all())


class InstallmentLifecycleTests(unittest.TestCase):
    def test_installments_after_closure_are_cancelled(self):
        plan, sched = Buf(), Buf()
        start = dt.date(2026, 1, 1)
        close = start + dt.timedelta(days=40)
        _emit_plan_v2(
            np.random.default_rng(7), plan, sched, "PLN1",
            SimpleNamespace(account_id="ACC1"), "CUS1",
            {"transaction_id": "TXN1", "date": start, "amount_usd": 300.0},
            1, close,
        )
        rows = sched.frame()
        after_close = rows[rows.due_date > close]
        self.assertGreater(len(after_close), 0)
        self.assertTrue((after_close.status == "CANCELLED").all())
        self.assertTrue((after_close.paid_amount == 0).all())
        self.assertTrue((after_close.paid_date.isna()).all())
        self.assertEqual(plan.frame().iloc[0].plan_status, "CANCELLED")


def _plan_dims() -> dict:
    """Minimal dimensions that drive one open and one planned-closed account."""
    merchants = pd.DataFrame([
        dict(merchant_id=f"M{i}", merchant_name=f"merchant {i}", mcc=code,
             category=category, geo_id=1, is_international=False,
             is_recurring=False, popularity_weight=1.0 / len(C.MCC_CATALOG))
        for i, (code, _desc, category, *_rest) in enumerate(C.MCC_CATALOG)
    ])
    customers = pd.DataFrame([
        dict(customer_id="C_OPEN", customer_segment="AFFLUENT_POWER", home_geo_id=1),
        dict(customer_id="C_CLOSE", customer_segment="AFFLUENT_POWER", home_geo_id=1),
    ])
    product_id = C.PRODUCTS[0]["product_id"]
    accounts = pd.DataFrame([
        dict(account_id="A_OPEN", customer_id="C_OPEN", product_id=product_id,
             open_date=dt.date(2023, 1, 1), planned_close_date=None,
             credit_limit=60000.0, apr_purchase=0.22, statement_day=5,
             foreign_txn_fee=0.0),
        dict(account_id="A_CLOSE", customer_id="C_CLOSE", product_id=product_id,
             open_date=dt.date(2023, 1, 1), planned_close_date=dt.date(2026, 6, 30),
             credit_limit=60000.0, apr_purchase=0.22, statement_day=5,
             foreign_txn_fee=0.0),
    ])
    cards = pd.DataFrame([
        dict(account_id="A_OPEN", card_id="K_OPEN",
             issue_date=dt.date(2023, 1, 1), replaced_date=None),
        dict(account_id="A_CLOSE", card_id="K_CLOSE",
             issue_date=dt.date(2023, 1, 1), replaced_date=None),
    ])
    return dict(dim_account=accounts, dim_customer=customers, dim_merchant=merchants,
                dim_channel=pd.DataFrame(C.CHANNELS), dim_card=cards)


class SimulatedPlanCloseDateTests(unittest.TestCase):
    """An account without a planned close keeps its post-period plans open."""

    @classmethod
    def setUpClass(cls):
        out = facts.simulate_v2(_plan_dims())
        cls.plan = out["plan"].frame()
        cls.sched = out["sched"].frame()

    def test_open_account_plans_are_not_cancelled_at_period_end(self):
        open_plans = self.plan[self.plan.account_id == "A_OPEN"]
        self.assertGreater(len(open_plans), 0)
        self.assertFalse((open_plans.plan_status == "CANCELLED").any())
        open_sched = self.sched[self.sched.account_id == "A_OPEN"]
        beyond = open_sched[open_sched.due_date > C.PERIOD_END]
        self.assertGreater(len(beyond), 0)
        self.assertFalse((beyond.status == "CANCELLED").any())

    def test_planned_close_still_cancels_only_installments_after_close(self):
        close = dt.date(2026, 6, 30)
        closed_sched = self.sched[self.sched.account_id == "A_CLOSE"]
        after = closed_sched[closed_sched.due_date > close]
        self.assertGreater(len(after), 0)
        self.assertTrue((after.status == "CANCELLED").all())
        within = closed_sched[closed_sched.due_date <= close]
        self.assertFalse((within.status == "CANCELLED").any())


class RawConnectivityTests(unittest.TestCase):
    def test_voucher_activity_uses_parent_voucher_id(self):
        vouchers = pd.DataFrame([dict(
            ID="VCH-TXN1", PURCHASE_DATE="2026-01-01", STATUS="REDEEM_CONFIRMED",
            TRANSACTION_IDENTIFIER="CC1", PAID_AMOUNT="10.00",
            hashed_VOUCHER_NUMBER="v", hashed_ORIGINAL_ID="o",
        )])
        activity = raw_mirror.build_voucher_activity(vouchers, np.random.default_rng(7))
        self.assertEqual(len(activity), 3)
        self.assertEqual(set(activity.VOUCHER_ID), {"VCH-TXN1"})

    def test_purchase_schedule_and_activity_share_transaction_id(self):
        plan = pd.DataFrame([dict(
            plan_id="PLN1", transaction_id="TXN1", plan_start_date="2026-01-01",
        )])
        sched = pd.DataFrame([dict(
            schedule_id="ISC1", plan_id="PLN1", account_id="ACC1", installment_number=1,
            due_date="2026-02-01", amount_due=100.0, status="PAID", paid_date="2026-01-30",
            paid_amount=100.0, interest_fee_component=0.0, principal_component=100.0,
        )])
        rng = np.random.default_rng(7)
        schedule_raw = raw_mirror.build_purchase_schedule(sched, plan, rng)
        activity_raw = raw_mirror.build_purchase_activity(sched, rng, plan)
        self.assertEqual(set(schedule_raw.PURCHASE_ID), {"TXN1"})
        self.assertEqual(set(activity_raw.PURCHASE_ID), {"TXN1"})
        self.assertEqual(set(activity_raw.PURCHASE_SCHEDULE_ID), {"ISC1"})


if __name__ == "__main__":
    unittest.main()
