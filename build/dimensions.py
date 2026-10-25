"""Dimension table generation: date, location, product, channel, merchant,
customer, account and card."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C
from .utils import (daterange, mask_id, rng_for, synthetic_phone, synthetic_ssn,
                    us_holidays)

# --------------------------------------------------------------------------
# dim_date
# --------------------------------------------------------------------------
def build_dim_date() -> pd.DataFrame:
    rows = []
    start = dt.date(2025, 1, 1)
    end = dt.date(2029, 12, 31)   # covers instalments that run past the fact window
    for d in daterange(start, end):
        rows.append(dict(
            date_key=int(d.strftime("%Y%m%d")),
            date=d,
            year=d.year,
            quarter=(d.month - 1) // 3 + 1,
            month=d.month,
            month_name=d.strftime("%B"),
            month_short=d.strftime("%b"),
            year_month=d.strftime("%Y-%m"),
            day_of_month=d.day,
            day_of_week=d.isoweekday(),
            day_name=d.strftime("%A"),
            day_short=d.strftime("%a"),
            iso_week=d.isocalendar().week,
            is_weekend=d.weekday() >= 5,
            is_holiday=d in us_holidays(d.year),
            is_month_start=d.day == 1,
            days_in_month=(dt.date(d.year + (d.month // 12), d.month % 12 + 1, 1) - dt.timedelta(days=1)).day,
            is_month_end=(d + dt.timedelta(days=1)).month != d.month,
            fiscal_year=d.year if d.month >= 10 else d.year - 1,
            fiscal_quarter=((d.month - 10) % 12) // 3 + 1,
        ))
    df = pd.DataFrame(rows)
    df["date_key"] = df["date_key"].astype("int32")
    return df


# --------------------------------------------------------------------------
# dim_location
# --------------------------------------------------------------------------
def build_dim_location() -> pd.DataFrame:
    rng = rng_for("dim_location")
    rows = []
    for m in C.METROS:
        for (city, st, prefix, lat, lon) in m["cities"]:
            for i in range(3):
                z = f"{prefix}{int(rng.integers(1, 100)):02d}"
                rows.append(dict(
                    geo_id=None,
                    city=city,
                    state_code=st,
                    zip_code=z,
                    metro=m["metro"],
                    cbsa_code=m["cbsa"],
                    region=m["region"],
                    country_code="US",
                    latitude=round(lat + float(rng.normal(0, 0.03)), 5),
                    longitude=round(lon + float(rng.normal(0, 0.03)), 5),
                    timezone=C.STATE_TZ.get(st, "America/New_York"),
                    is_metro_core=True,
                ))
    for (city, st, cc, z, tz, lat, lon) in C.INTL_MERCHANTS_GEO:
        rows.append(dict(
            geo_id=None, city=city, state_code=st, zip_code=z,
            metro=f"International - {city}", cbsa_code=None, region="International",
            country_code=cc, latitude=lat, longitude=lon, timezone=tz,
            is_metro_core=False,
        ))
    df = pd.DataFrame(rows).drop_duplicates(subset=["city", "state_code", "zip_code"])
    df = df.sort_values(["state_code", "city", "zip_code"]).reset_index(drop=True)
    df["geo_id"] = [f"LOC{i:05d}" for i in range(1, len(df) + 1)]
    return df[["geo_id", "city", "state_code", "zip_code", "metro", "cbsa_code", "region",
               "country_code", "latitude", "longitude", "timezone", "is_metro_core"]]


# --------------------------------------------------------------------------
# dim_product / dim_channel
# --------------------------------------------------------------------------
def build_dim_product() -> pd.DataFrame:
    df = pd.DataFrame(C.PRODUCTS).drop(columns=["share"])
    df.insert(0, "product_sk", range(1, len(df) + 1))
    return df


def build_dim_channel() -> pd.DataFrame:
    df = pd.DataFrame(C.CHANNELS).drop(columns=["base_share"])
    df.insert(0, "channel_sk", range(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------
# dim_merchant
# --------------------------------------------------------------------------
def build_dim_merchant(geo: pd.DataFrame) -> pd.DataFrame:
    rng = rng_for("dim_merchant")
    mcc_weight = np.array([m[7] for m in C.MCC_CATALOG], dtype=float)
    mcc_weight = mcc_weight / mcc_weight.sum()

    us_geo = geo[geo.country_code == "US"].sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
    intl_geo = geo[geo.country_code != "US"].set_index(["city", "country_code"])["geo_id"].to_dict()
    rows = []
    total = C.N_MERCHANTS
    counts = np.maximum(4, np.round(mcc_weight * total).astype(int))

    brand_seq: dict[str, int] = {}
    mid = 1
    for (code, desc, category, med, sigma, online, wknd, weight, foreign), n in zip(C.MCC_CATALOG, counts):
        brands = C.MERCHANT_BRANDS.get(category) or C.MERCHANT_BRANDS["Retail"]
        for k in range(int(n)):
            brand = brands[int(rng.integers(0, len(brands)))]
            brand_seq[brand] = brand_seq.get(brand, 0) + 1
            name = f"{brand} #{brand_seq[brand]:05d}"
            is_intl = rng.random() < 0.035 and foreign > 0.02
            if is_intl:
                city, st, cc, z, tz, lat, lon = C.INTL_MERCHANTS_GEO[int(rng.integers(0, len(C.INTL_MERCHANTS_GEO)))]
                country = cc
                region = "International"
                geo_id = intl_geo.get((city, cc))
            else:
                g = us_geo.iloc[int(rng.integers(0, len(us_geo)))]
                city, st, z = g.city, g.state_code, g.zip_code
                geo_id, country, region = g.geo_id, "US", g.region
            rows.append(dict(
                merchant_id=f"M{mid:06d}",
                merchant_name=name,
                merchant_brand=brand,
                mcc=code,
                mcc_description=desc,
                category=category,
                geo_id=geo_id,
                city=city,
                state_code=st,
                zip_code=z,
                country_code=country,
                region=region,
                channel_primary="ONLINE" if rng.random() < online else "IN_STORE",
                is_recurring=code in {"4899", "4814", "4900", "5968", "5815", "5816", "7997"},
                merchant_size_tier=str(rng.choice(["SMALL", "MID", "LARGE"], p=[0.55, 0.33, 0.12])),
                popularity_weight=round(float(rng.pareto(1.8) + 0.15), 4),
                is_international=bool(is_intl),
            ))
            mid += 1

    df = pd.DataFrame(rows)
    # keep popularity bounded so a handful of merchants do not dominate
    df["popularity_weight"] = df["popularity_weight"].clip(0.15, 12.0)
    df["popularity_weight"] = df["popularity_weight"] / df.groupby("mcc")["popularity_weight"].transform("sum")
    return df


# --------------------------------------------------------------------------
# dim_customer + dim_account + dim_card
# --------------------------------------------------------------------------
RISK_BANDS = [(780, 851, "PRIME_PLUS"), (720, 780, "PRIME"), (660, 720, "NEAR_PRIME"),
              (600, 660, "SUBPRIME"), (300, 600, "DEEP_SUBPRIME")]

# Income is drawn from US occupation priors but rescaled so the book's median
# matches the source median treated 1:1 as USD.
INCOME_SCALE = C.SOURCE_INCOME_MEDIAN / float(np.median([o[2] for o in C.OCCUPATIONS]))


def _assign_segments(customers: pd.DataFrame, tiers: list[str]) -> pd.DataFrame:
    """Behavioural segmentation by propensity, allocated on exact target shares.

    Propensity blends income, credit score, product tier and age. Customers are
    ranked and cut into the configured shares so the population mix is exact and
    the median transactions-per-month lands on the CORE_MAINSTREAM value.
    """
    def z(s):
        sd = float(s.std(ddof=0)) or 1.0
        return (s - float(s.mean())) / sd

    tier_map = {"SECURED": -1.2, "STANDARD": -0.4, "SIGNATURE": 0.3,
                "PLATINUM": 0.7, "GOLD": 1.0, "BUSINESS": 0.6}
    tscore = pd.Series([tier_map.get(t, 0.0) for t in tiers], index=customers.index)

    prop = (0.45 * z(customers.income_annual)
            + 0.30 * z(customers.credit_score)
            + 0.15 * tscore
            + 0.10 * z(customers.age_years))
    # 0 = most valuable customer
    q = 1.0 - prop.rank(pct=True, method="first")

    shares = [s["share"] for s in C.CUSTOMER_SEGMENTS]
    cum = np.cumsum(shares)[:-1]
    idx = np.searchsorted(cum, q.to_numpy(), side="right")
    names = [s["segment"] for s in C.CUSTOMER_SEGMENTS]
    medians = [s["median_txn"] for s in C.CUSTOMER_SEGMENTS]

    customers["customer_segment"] = [names[i] for i in idx]
    customers["expected_txn_per_month"] = [medians[i] for i in idx]
    return customers


def _risk_tier(score: int) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= score < hi:
            return name
    return "DEEP_SUBPRIME"


def _trunc_normal(rng, mean, sd, lo, hi, n):
    vals = rng.normal(mean, sd, n)
    vals = np.clip(vals, lo, hi)
    return vals


def build_customers_accounts_cards(geo: pd.DataFrame):
    rng = rng_for("customers")
    products = pd.DataFrame(C.PRODUCTS)
    prod_p = products["share"].to_numpy(float)
    prod_p = prod_p / prod_p.sum()
    prod_idx = rng.choice(len(products), size=C.N_CUSTOMERS, p=prod_p)

    metro_w = np.array([m["weight"] for m in C.METROS], float)
    metro_w = metro_w / metro_w.sum()

    geo_by_metro: dict[str, list[int]] = {}
    for i, r in geo.iterrows():
        geo_by_metro.setdefault(r.metro, []).append(i)

    cust_rows = []
    cust_tier = []
    acct_rows = []
    card_rows = []
    acct_no = 1
    card_no = 1

    for i in range(C.N_CUSTOMERS):
        p = products.iloc[prod_idx[i]]
        product_id = p.product_id
        cust_tier.append(p["tier"])

        # credit score consistent with product tier
        lo = max(300, int(p.min_credit_score) - 40)
        hi = min(850, int(p.min_credit_score) + 130)
        score = int(round(float(_trunc_normal(rng, (lo + hi) / 2, (hi - lo) / 5.5, lo, hi, 1)[0])))

        gender = "F" if rng.random() < 0.505 else "M"
        first = C.FIRST_NAMES_F[int(rng.integers(0, len(C.FIRST_NAMES_F)))] if gender == "F" \
            else C.FIRST_NAMES_M[int(rng.integers(0, len(C.FIRST_NAMES_M)))]
        last = C.SURNAMES[int(rng.integers(0, len(C.SURNAMES)))]

        age = int(round(float(np.clip(rng.beta(2.4, 3.2) * 62 + 19, 19, 84))))
        dob = dt.date(2026, 9, 30) - dt.timedelta(days=age * 365 + int(rng.integers(0, 365)))

        occ, industry, base_income = C.OCCUPATIONS[int(rng.integers(0, len(C.OCCUPATIONS)))]
        if occ == "Student":
            age = int(min(age, 26))
            dob = dt.date(2026, 9, 30) - dt.timedelta(days=age * 365 + int(rng.integers(0, 365)))
        # 1:1 PHP->USD: anchor the book to the source income median rather than
        # to US salary levels.
        income = base_income * INCOME_SCALE * float(np.clip(rng.normal(1.0, 0.22), 0.55, 2.4))
        income = float(np.clip(income, 12000.0, C.SOURCE_INCOME_MAX))
        income = round(income / 1000) * 1000

        emp_status = ("STUDENT" if occ == "Student" else "RETIRED" if occ == "Retired"
                      else "SELF_EMPLOYED" if occ == "Self-Employed Consultant" else
                      str(rng.choice(C.EMPLOYMENT_STATUS, p=[0.78, 0.09, 0.06, 0.04, 0.03])))
        housing = str(rng.choice(C.HOUSING_STATUS, p=[0.38, 0.36, 0.19, 0.07]))
        education = str(rng.choice(C.EDUCATION, p=[0.20, 0.24, 0.09, 0.30, 0.14, 0.03]))
        marital = str(rng.choice(C.MARITAL_STATUS, p=[0.44, 0.42, 0.09, 0.03, 0.02]))

        mname = C.METROS[int(rng.choice(len(C.METROS), p=metro_w))]
        gidx = int(rng.choice(geo_by_metro[mname["metro"]]))
        g = geo.loc[gidx]

        cust_id = f"CUST{i + 1:06d}"
        # tenure: onboarding between 2022-06 and 2026-08, weighted toward earlier
        max_tenure = (C.PERIOD_END - dt.date(2022, 6, 1)).days
        tenure = int(max_tenure * float(rng.beta(1.35, 2.1)))
        onboard = C.PERIOD_END - dt.timedelta(days=tenure)
        onboard = min(onboard, dt.date(2026, 8, 15))

        churned = rng.random() < 0.055
        churn_date = None
        if churned:
            possible = (C.PERIOD_END - max(onboard + dt.timedelta(days=60), C.PERIOD_START)).days
            if possible > 30:
                churn_date = max(onboard + dt.timedelta(days=60), C.PERIOD_START) + \
                    dt.timedelta(days=int(rng.integers(0, possible)))
            else:
                churned = False   # too new to churn: keep active with a real date

        cust_rows.append(dict(
            customer_id=cust_id,
            first_name=first,
            last_name=last,
            full_name=f"{first} {last}",
            date_of_birth=dob,
            age_years=age,
            gender=gender,
            email=f"{first.lower()}.{last.lower()}{i + 1}@{str(rng.choice(['gmail.com','yahoo.com','outlook.com','icloud.com','proton.me']))}",
            phone=f"+1-{int(rng.choice([201,212,305,312,404,415,512,602,617,704,713,718,786,801,813,917,919,972,206]))}-"
                  f"{200 + ((i // 9000) % 800):03d}-{1000 + (i % 9000):04d}",
            ssn_last4=synthetic_ssn(rng)[-4:],
            id_type=str(rng.choice(C.ID_TYPES, p=[0.62, 0.16, 0.12, 0.06, 0.04])),
            id_number_masked=mask_id(f"{int(rng.integers(10**7, 10**8))}", 4),
            home_geo_id=g.geo_id,
            address_line1=f"{int(rng.integers(100, 9999))} {str(rng.choice(['Maple','Oak','Cedar','Pine','Elm','Sunset','Lakeview','Ridge','Park','Hill','Washington','Lincoln','Highland','River','Chestnut']))} "
                          f"{str(rng.choice(['St','Ave','Blvd','Dr','Ln','Ct','Way','Rd']))}"
                          f"{', Apt ' + str(int(rng.integers(1, 40))) if rng.random() < 0.28 else ''}",
            city=g.city,
            state_code=g.state_code,
            zip_code=g.zip_code,
            metro=g.metro,
            region=g.region,
            income_annual=income,
            employment_status=emp_status,
            employer=str(rng.choice(C.EMPLOYERS)) if emp_status in ("EMPLOYED", "SELF_EMPLOYED") else None,
            occupation=occ,
            industry=industry,
            education=education,
            marital_status=marital,
            housing_status=housing,
            credit_score=score,
            risk_tier=_risk_tier(score),
            acquisition_channel=str(rng.choice(C.ACQUISITION_CHANNELS)),
            onboarding_date=onboard,
            kyc_status="VERIFIED" if rng.random() > 0.02 else "PENDING",
            is_email_verified=bool(rng.random() < 0.94),
            is_phone_verified=bool(rng.random() < 0.91),
            marketing_opt_in=bool(rng.random() < 0.62),
            customer_status="CHURNED" if churned else "ACTIVE",
            churn_date=churn_date,
        ))

        # ---- accounts: primary, plus occasional second product
        n_accounts = 1
        if score >= 690 and rng.random() < 0.13:
            n_accounts = 2
        for a in range(n_accounts):
            if a == 1:
                pool = [x for x in C.PRODUCTS if x["product_id"] != product_id
                        and x["min_credit_score"] <= score and not x["is_secured"]]
                if not pool:
                    continue
                p2 = pool[int(rng.integers(0, len(pool)))]
            else:
                p2 = p

            util_target = float(np.clip(rng.beta(2.0, 3.0) * 0.55 + 0.05, 0.03, 0.85))
            limit = p2["min_limit"] + (p2["max_limit"] - p2["min_limit"]) * float(rng.beta(2.0, 3.5))
            limit *= float(np.clip(0.55 + income / 90000, 0.6, 1.8))
            limit *= C.CREDIT_LIMIT_SCALE
            limit = int(round(limit / 100) * 100)
            limit = int(np.clip(limit, p2["min_limit"] * C.CREDIT_LIMIT_SCALE,
                                p2["max_limit"] * C.CREDIT_LIMIT_SCALE))

            open_date = max(onboard + dt.timedelta(days=int(rng.integers(1, 12))), dt.date(2022, 6, 15))
            stmt_day = int(rng.integers(1, 29))

            acct = f"ACC{acct_no:07d}"
            acct_no += 1
            acct_rows.append(dict(
                account_id=acct,
                customer_id=cust_id,
                product_id=p2["product_id"],
                product_name=p2["product_name"],
                network=p2["network"],
                open_date=open_date,
                credit_limit=limit,
                initial_credit_limit=limit,
                apr_purchase=p2["apr_purchase"],
                apr_cash=p2["apr_cash"],
                foreign_txn_fee=p2["foreign_txn_fee"],
                rewards_rate=p2["rewards_rate"],
                account_status="ACTIVE",
                is_primary=bool(a == 0),
                is_secured=bool(p2["is_secured"]),
                is_business=bool(p2["is_business"]),
                statement_day=stmt_day,
                payment_due_days_after_statement=25,
                planned_close_date=churn_date if churned else None,
                target_utilization=round(util_target, 4),
                acquisition_cost=round(float(rng.uniform(60, 320)), 2),
            ))

            # ---- cards
            cards_for_acct = []
            n_cards = 1
            if rng.random() < 0.18:
                n_cards = 2
            replace_prob = 0.06
            for c in range(n_cards):
                if c == 0:
                    issue = open_date
                else:
                    prev = cards_for_acct[-1]
                    issue = prev["replaced_date"] or (open_date + dt.timedelta(days=int(rng.integers(120, 500))))
                if issue > C.PERIOD_END:
                    continue
                if churned and churn_date is not None and issue > churn_date:
                    continue
                ctype = "VIRTUAL" if (c == 1 and rng.random() < 0.55) else "PHYSICAL"
                replaced = rng.random() < replace_prob
                replaced_date = issue + dt.timedelta(days=int(rng.integers(300, 800))) if replaced else None
                if replaced_date is not None and replaced_date > C.PERIOD_END:
                    replaced_date = None
                if churned and churn_date is not None and replaced_date is not None and replaced_date > churn_date:
                    replaced_date = None
                rec = dict(
                    card_id=f"CRD{card_no:08d}",
                    account_id=acct,
                    customer_id=cust_id,
                    card_type=ctype,
                    network=p2["network"],
                    last4=f"{int(rng.integers(1000, 9999))}",
                    expiry_month=(issue.month + int(rng.integers(0, 12)) - 1) % 12 + 1,
                    expiry_year=max(issue.year + 4,
                                    C.PERIOD_END.year + 1 if c == n_cards - 1 else issue.year + 4),
                    issue_date=issue,
                    replaced_date=replaced_date,
                    card_status="REPLACED" if replaced_date else "ACTIVE",
                    replacement_reason=(str(rng.choice(["LOST", "DAMAGED", "FRAUD"]))
                                        if replaced_date else None),
                    is_contactless=bool(ctype == "PHYSICAL" and rng.random() < 0.93),
                    is_virtual=bool(ctype == "VIRTUAL"),
                )
                cards_for_acct.append(rec)
                card_rows.append(rec)
                card_no += 1

            # a replaced card must have a successor, otherwise it stays valid
            if cards_for_acct and cards_for_acct[-1]["replaced_date"] is not None:
                cards_for_acct[-1]["replaced_date"] = None
                cards_for_acct[-1]["card_status"] = "ACTIVE"
                cards_for_acct[-1]["replacement_reason"] = None

    customers = pd.DataFrame(cust_rows)
    customers = _assign_segments(customers, cust_tier)
    accounts = pd.DataFrame(acct_rows)
    cards = pd.DataFrame(card_rows)
    return customers, accounts, cards


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
def build_all():
    dim_date = build_dim_date()
    dim_location = build_dim_location()
    dim_product = build_dim_product()
    dim_channel = build_dim_channel()
    dim_merchant = build_dim_merchant(dim_location)
    dim_customer, dim_account, dim_card = build_customers_accounts_cards(dim_location)
    return dict(
        dim_date=dim_date,
        dim_location=dim_location,
        dim_product=dim_product,
        dim_channel=dim_channel,
        dim_merchant=dim_merchant,
        dim_customer=dim_customer,
        dim_account=dim_account,
        dim_card=dim_card,
    )
