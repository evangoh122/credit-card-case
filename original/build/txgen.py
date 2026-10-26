"""Vectorised transaction generation.

The sequential statement-cycle simulator needs transaction *dates* and *amounts*
one cycle at a time, but drawing them one Python call at a time does not scale to
millions of rows. This module draws a whole cycle in one pass with NumPy while
preserving the same distributions (MCC-driven amounts, channel mix, time-of-day,
declines, refunds, international share).
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import config as C

_MCC_MEDIAN = {m[0]: m[3] for m in C.MCC_CATALOG}
_MCC_SIGMA = {m[0]: m[4] for m in C.MCC_CATALOG}
_MCC_ONLINE = {m[0]: m[5] for m in C.MCC_CATALOG}
_MCC_FOREIGN = {m[0]: m[8] for m in C.MCC_CATALOG}
_MCC_CAT = {m[0]: m[2] for m in C.MCC_CATALOG}
_MCC_CODE = [m[0] for m in C.MCC_CATALOG]

# ticket calibration coefficients live in config so both layers agree
AMOUNT_A = C.AMOUNT_A
AMOUNT_B = C.AMOUNT_B


class TxContext:
    """Precomputed lookup arrays shared by every cycle."""

    def __init__(self, merch, channels):
        self.merch = merch.reset_index(drop=True)
        idx_of = {c: i for i, c in enumerate(_MCC_CODE)}
        self.merch_mcc_idx = np.array([idx_of[c] for c in self.merch.mcc], dtype=np.int32)

        # P(merchant) = P(mcc) * P(merchant | mcc); popularity already sums to 1
        w = np.array([m[7] for m in C.MCC_CATALOG], float)
        w = w / w.sum()
        p = w[self.merch_mcc_idx] * self.merch.popularity_weight.to_numpy(float)
        self.merch_p = p / p.sum()
        self.merch_cdf = np.cumsum(self.merch_p)
        self.merch_cdf[-1] = 1.0

        self.merch_ids = self.merch.merchant_id.to_numpy()
        self.merch_names = self.merch.merchant_name.to_numpy()
        self.merch_geo = self.merch.geo_id.to_numpy()
        self.merch_intl = self.merch.is_international.to_numpy(bool)
        self.merch_cat = self.merch.category.to_numpy()
        self.merch_recurring = self.merch.is_recurring.to_numpy(bool)

        self.mcc_code = np.array(_MCC_CODE)
        self.mcc_median = np.array([_MCC_MEDIAN[c] for c in _MCC_CODE], float)
        self.mcc_sigma = np.array([_MCC_SIGMA[c] for c in _MCC_CODE], float)
        self.mcc_online = np.array([_MCC_ONLINE[c] for c in _MCC_CODE], float)
        self.mcc_foreign = np.array([_MCC_FOREIGN[c] for c in _MCC_CODE], float)

        groups = sorted({c for c in _MCC_CAT.values()})
        gidx = {g: i for i, g in enumerate(groups)}
        self.group_of_mcc = np.array([gidx[_MCC_CAT[c]] for c in _MCC_CODE], np.int32)
        self.hour_cdf = np.vstack([_hour_cdf(g) for g in groups])

        # channels
        ch = channels.reset_index(drop=True)
        on = ch[ch.channel_group.isin(["ONLINE", "MOBILE_APP", "PHONE_ORDER"])]
        off = ch[ch.channel_group == "IN_STORE"]
        self.on_idx = on.index.to_numpy()
        self.off_idx = off.index.to_numpy()
        on_p = np.array([0.76, 0.20, 0.04])
        off_p = np.array([0.52, 0.42, 0.06])
        self.on_p = on_p / on_p.sum()
        self.off_p = off_p / off_p.sum()
        self.ch_id = ch.channel_id.to_numpy()
        self.ch_group = ch.channel_group.to_numpy()
        self.ch_entry = ch.entry_mode.to_numpy()
        self.ch_present = ch.is_card_present.to_numpy(bool)


def _hour_cdf(group: str) -> np.ndarray:
    w = np.ones(24, float)
    if group == "Dining":
        w[:7] = 0.02; w[11:14] = 1.7; w[17:21] = 1.9; w[21:] = 1.1
    elif group == "Grocery":
        w[:7] = 0.05; w[9:20] = 1.4; w[20:] = 0.5
    elif group == "Fuel":
        w[:5] = 0.05; w[6:20] = 1.3; w[20:] = 0.4
    elif group in ("Travel", "Transport"):
        w[:5] = 0.25; w[5:23] = 1.1
    elif group in ("Digital", "E-Commerce", "Telecom"):
        w[:7] = 0.15; w[18:24] = 1.7; w[12:18] = 0.8
    elif group == "Entertainment":
        w[:9] = 0.05; w[17:24] = 1.8
    elif group == "Utilities":
        w[:6] = 0.05; w[8:21] = 1.0
    else:
        w[:7] = 0.05; w[8:21] = 1.2; w[21:] = 0.6
    w = w / w.sum()
    return np.cumsum(w)


def generate_cycle(ctx: TxContext, rng: np.random.Generator, days: list[dt.date],
                   weights: np.ndarray, n: int, limit: float, running: float,
                   account_age_days: int):
    """Draw one statement cycle of transactions. Returns a dict of arrays."""
    if n <= 0:
        return None, max(0.0, limit - running)

    day_idx = rng.choice(len(days), size=n, p=weights)

    mi = np.minimum(np.searchsorted(ctx.merch_cdf, rng.random(n), side="right"),
                    len(ctx.merch_cdf) - 1)
    mcc_i = ctx.merch_mcc_idx[mi]
    online = rng.random(n) < ctx.mcc_online[mcc_i]
    ci = np.where(online,
                  ctx.on_idx[rng.choice(len(ctx.on_idx), size=n, p=ctx.on_p)],
                  ctx.off_idx[rng.choice(len(ctx.off_idx), size=n, p=ctx.off_p)])

    grp = ctx.group_of_mcc[mcc_i]
    u = rng.random(n)[:, None]
    hour = np.clip((u > ctx.hour_cdf[grp]).sum(axis=1), 0, 23)
    minute = rng.integers(0, 60, n)
    second = rng.integers(0, 60, n)

    raw = rng.lognormal(np.log(ctx.mcc_median[mcc_i]), ctx.mcc_sigma[mcc_i] * 0.75)
    amount = np.round(np.clip(AMOUNT_A * raw ** AMOUNT_B, 1.0, C.US_TICKET_MAX), 2)

    day_ts = np.array([np.datetime64(d) for d in days], dtype="datetime64[s]")
    ts = (day_ts[day_idx]
          + hour.astype("timedelta64[h]")
          + minute.astype("timedelta64[m]")
          + second.astype("timedelta64[s]"))

    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    mi, mcc_i, ci = mi[order], mcc_i[order], ci[order]
    amount = amount[order]
    hour = hour[order]
    online = online[order]

    # ---- authorisation: cumulative spend cannot exceed available credit ------
    avail = max(0.0, limit - running)
    approved = np.ones(n, bool)
    for _ in range(3):
        cum = np.cumsum(np.where(approved, amount, 0.0))
        over = cum > avail * 0.995
        if not over.any():
            break
        approved &= ~over

    status = np.where(approved, "APPROVED", "DECLINED")
    decline_code = np.where(approved, "", "CREDIT_LIMIT_EXCEEDED")

    # random declines
    r = rng.random(n)
    dnh = approved & (r < 0.011)
    if dnh.any():
        status[dnh] = "DECLINED"
        decline_code[dnh] = np.where(rng.random(dnh.sum()) < 0.55, "DO_NOT_HONOR", "INVALID_CARD")

    risky = (status == "APPROVED") & (rng.random(n) < 0.004)
    if risky.any():
        status[risky] = "DECLINED"
        decline_code[risky] = "SUSPECTED_FRAUD"

    reversed_ = (status == "APPROVED") & (rng.random(n) < 0.012)
    status[reversed_] = "REVERSED"

    # international status is a property of the merchant, not a separate draw
    is_intl = ctx.merch_intl[mi]
    fx = np.ones(n)
    orig = amount.copy()
    currency = np.full(n, "USD", dtype=object)
    if is_intl.any():
        codes = np.array(["CAD", "GBP", "EUR", "MXN", "JPY", "AUD"], dtype=object)
        rates = np.array([1.36, 0.79, 0.92, 17.4, 148.0, 1.52])
        pick = rng.integers(0, len(codes), int(is_intl.sum()))
        currency[is_intl] = codes[pick]
        fx[is_intl] = 1.0 / rates[pick]
        orig[is_intl] = np.round(amount[is_intl] * rates[pick], 2)

    cat = ctx.merch_cat[mi]
    risk = (1.55 * (amount > 500)
            + 2.30 * (~ctx.ch_present[ci])
            + 2.60 * is_intl
            + 1.10 * (hour <= 5)
            + 1.40 * np.isin(cat, ["Entertainment", "Electronics", "Digital", "E-Commerce"])
            + 0.55 * (account_age_days < 60)
            + 0.35 * (status == "DECLINED")
            + rng.normal(0, 0.8, n))
    score = np.round(1.0 / (1.0 + np.exp(-(risk - 2.6))), 4)

    # ---- fraud event, decided BEFORE final settlement state ----------------
    flag = (status == "APPROVED") & (rng.random(n) < (0.0013 + 0.0050 * score))
    decision = np.full(n, "", dtype=object)
    case = np.full(n, "", dtype=object)
    if flag.any():
        k = int(flag.sum())
        decision[flag] = rng.choice(["BLOCK", "STEP_UP", "ALLOW"], size=k, p=[0.40, 0.38, 0.22])
        u = rng.random(k)
        d = decision[flag]
        confirmed = np.where(d == "ALLOW", u < 0.60,
                     np.where(d == "STEP_UP", u < 0.40, u < 0.25))
        case[flag] = np.where(confirmed, "CONFIRMED_FRAUD",
                              np.where(u < 0.75, "FALSE_POSITIVE", "OPEN"))

    # a blocked attempt never settles
    blocked = flag & (decision == "BLOCK")
    status[blocked] = "DECLINED"
    decline_code[blocked] = "SUSPECTED_FRAUD"

    reversed_ = (status == "APPROVED") & (rng.random(n) < 0.012)
    status[reversed_] = "REVERSED"

    # ---- refunds: duplicate a subset of settled purchases, linked to parent ---
    refund_mask = (status == "APPROVED") & (rng.random(n) < 0.022) & (~is_intl)
    out = dict(
        merchant_idx=mi, mcc=ctx.mcc_code[mcc_i], category=cat, channel_idx=ci,
        ts=ts, amount_usd=amount, currency_code=currency, original_amount=orig,
        fx_rate=fx, status=status, decline_code=decline_code,
        is_international=is_intl, fraud_score=score,
        fraud_flag=flag, fraud_decision=decision, fraud_case_status=case,
        parent_pos=np.full(n, -1, dtype=np.int64),
    )
    if refund_mask.any():
        k = int(refund_mask.sum())
        src = np.where(refund_mask)[0]
        r_amount = -np.round(amount[src] * rng.uniform(0.25, 1.0, k), 2)
        out["merchant_idx"] = np.concatenate([mi, mi[src]])
        out["mcc"] = np.concatenate([ctx.mcc_code[mcc_i], ctx.mcc_code[mcc_i[src]]])
        out["category"] = np.concatenate([cat, cat[src]])
        out["channel_idx"] = np.concatenate([ci, ci[src]])
        out["ts"] = np.concatenate([ts, ts[src]])
        out["amount_usd"] = np.concatenate([amount, r_amount])
        out["currency_code"] = np.concatenate([currency, np.full(k, "USD", dtype=object)])
        out["original_amount"] = np.concatenate([orig, r_amount])
        out["fx_rate"] = np.concatenate([fx, np.ones(k)])
        out["status"] = np.concatenate([status, np.full(k, "APPROVED", dtype=object)])
        out["decline_code"] = np.concatenate([decline_code, np.full(k, "", dtype=object)])
        out["is_international"] = np.concatenate([is_intl, np.zeros(k, bool)])
        out["fraud_score"] = np.concatenate([score, np.zeros(k)])
        out["fraud_flag"] = np.concatenate([flag, np.zeros(k, bool)])
        out["fraud_decision"] = np.concatenate([decision, np.full(k, "", dtype=object)])
        out["fraud_case_status"] = np.concatenate([case, np.full(k, "", dtype=object)])
        out["parent_pos"] = np.concatenate([out["parent_pos"], src.astype(np.int64)])

    out["transaction_type"] = np.where(out["amount_usd"] < 0, "REFUND", "PURCHASE").astype(object)

    spent = float(np.where(out["transaction_type"] == "PURCHASE",
                           np.where(out["status"] == "APPROVED", out["amount_usd"], 0.0), 0.0).sum())
    return out, max(0.0, limit - running - spent)
