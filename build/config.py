"""Configuration and reference catalogs for the synthetic US credit-card dataset.

All values are calibrated to plausible US card-issuer behaviour. The source
(PH BNPL) distributions were used as shape templates; amounts, geography,
identity and product constructs are re-expressed for a US credit-card book.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
# Output folder is overridable so a rebuild can be written alongside an earlier
# version without overwriting it:  set CC_OUT_DIR=credit_card_data_v2
OUT = ROOT / os.environ.get("CC_OUT_DIR", "credit_card_data")
DIM_DIR = OUT / "dim"
FACT_DIR = OUT / "fact"
RAW_DIR = OUT / "raw"
DOC_DIR = OUT / "docs"
SOURCE_DIR = ROOT / "tyme_data_download" / "tyme"

# --------------------------------------------------------------------------
# Reproducibility & period
# --------------------------------------------------------------------------
SEED = 20260924
N_CUSTOMERS = 50000
PERIOD_START = dt.date(2025, 9, 1)
PERIOD_END = dt.date(2026, 9, 30)
HISTORY_START = dt.date(2023, 1, 1)      # onboarding may predate the fact window
PAYMENT_HORIZON = dt.date(2026, 9, 30)   # payments never recorded after this

# --------------------------------------------------------------------------
# Currency convention
# --------------------------------------------------------------------------
# The source book is denominated in PHP. Per the brief, PHP amounts are carried
# over as USD at 1:1 (1000 PHP -> 1000 USD). No FX division is applied, so the
# synthetic ticket distribution is anchored to the source's own numeric
# distribution rather than to US market norms.
#
# Source purchase AMOUNT percentiles (PHP) used as the calibration anchors:
SOURCE_P50 = 2458.38
SOURCE_P25 = 1811.00
SOURCE_P75 = 3759.21
SOURCE_P95 = 6699.15
SOURCE_MIN = 1500.00
SOURCE_MAX = 24030.00

# Source purchasing frequency (bnpl_purchase: 4,241 purchases over 289 accounts
# across 5.2 months -> 2.82 purchases per account per month)
SOURCE_TXN_PER_ACCOUNT_MONTH = 2.82

# --------------------------------------------------------------------------
# Target behaviour (US card book)
# --------------------------------------------------------------------------
# Amounts are calibrated to US card norms rather than 1:1 to PHP: at 15
# transactions per month a 1:1 median ticket of $2,458 implies ~$37k of monthly
# spend per card, which is not a viable consumer book.
US_TICKET_MEDIAN = 85.0
US_TICKET_P95 = 600.0
US_TICKET_MAX = 15000.0

TARGET_MEDIAN_TXN_PER_ACCOUNT_MONTH = 15.0

# Merchant universe. At ~15M transactions a 727-merchant catalogue would give each
# merchant ~20k transactions, which is unrealistic; 20k merchants gives ~750 each.
N_MERCHANTS = 20000

# Month-to-month overdispersion: a monthly multiplier with median 1 widens the
# count distribution well beyond Poisson, and a rare spike month produces the
# 100+ transaction outliers.
TXN_MONTH_SIGMA = 0.60
SPIKE_MONTH_PROB = 0.015
SPIKE_MULT_RANGE = (2.5, 6.0)
MAX_TXN_PER_MONTH = 450

# Behavioural segments: share of customers and median transactions per month.
# The weighted median lands on the CORE_MAINSTREAM value of 15.
CUSTOMER_SEGMENTS = [
    dict(segment="AFFLUENT_POWER", share=0.10, median_txn=40.0, sigma=0.30,
         description="High-income premium cardholder, heavy everyday spender"),
    dict(segment="PRIME_ACTIVE", share=0.25, median_txn=20.0, sigma=0.30,
         description="Prime credit, multi-category active spender"),
    dict(segment="CORE_MAINSTREAM", share=0.35, median_txn=15.0, sigma=0.30,
         description="Mainstream prime/near-prime, regular usage"),
    dict(segment="LIGHT_USER", share=0.22, median_txn=6.0, sigma=0.35,
         description="Low usage, few categories, often carries a balance"),
    dict(segment="DORMANT", share=0.08, median_txn=1.5, sigma=0.45,
         description="Near-inactive, low spend, reactivation candidates"),
]

# Source credit facility (PHP -> USD 1:1)
SOURCE_LIMIT_MIN = 4500.0
SOURCE_LIMIT_MEDIAN = 10000.0
SOURCE_LIMIT_MAX = 49500.0

# The source "facility" financed a single 4-instalment plan at a time. With
# US-normal ticket sizes a 1x facility is already generous relative to monthly
# spend, so no scaling is applied.
CREDIT_LIMIT_SCALE = 1.0

# --------------------------------------------------------------------------
# Raw mirror sampling
# --------------------------------------------------------------------------
# At 50k customers the star schema carries ~11M transactions. The raw mirror is
# an operational sample rather than a full copy so the deliverable stays portable.
RAW_TXN_CAP = 150_000
RAW_PLAN_CAP = 30_000
RAW_PAYMENT_CAP = 60_000
RAW_FRAUD_CAP = 20_000

# Source self-reported income (PHP -> USD 1:1), median used as the anchor
SOURCE_INCOME_MEDIAN = 30000.0
SOURCE_INCOME_MAX = 470007.0

# --------------------------------------------------------------------------
# Card products
# --------------------------------------------------------------------------
PRODUCTS = [
    dict(product_id="PRD_CASHBACK_PLAT", product_name="Platinum Cashback",
         tier="PLATINUM", network="Visa", annual_fee=95.0, apr_purchase=0.2199,
         apr_cash=0.2799, foreign_txn_fee=0.0, rewards_rate=0.02, rewards_type="CASHBACK",
         min_credit_score=720, min_limit=8000, max_limit=35000, is_secured=False,
         is_business=False, share=0.22),
    dict(product_id="PRD_TRAVEL_SIGNATURE", product_name="Signature Travel Rewards",
         tier="SIGNATURE", network="Mastercard", annual_fee=0.0, apr_purchase=0.2399,
         apr_cash=0.2999, foreign_txn_fee=0.0, rewards_rate=0.015, rewards_type="POINTS",
         min_credit_score=690, min_limit=6000, max_limit=28000, is_secured=False,
         is_business=False, share=0.26),
    dict(product_id="PRD_SECURED_BUILD", product_name="Secured Build Card",
         tier="SECURED", network="Visa", annual_fee=0.0, apr_purchase=0.2699,
         apr_cash=0.2999, foreign_txn_fee=0.03, rewards_rate=0.0, rewards_type="NONE",
         min_credit_score=520, min_limit=4500, max_limit=9000, is_secured=True,
         is_business=False, share=0.16),
    dict(product_id="PRD_STUDENT_STARTER", product_name="Student Starter",
         tier="STANDARD", network="Mastercard", annual_fee=0.0, apr_purchase=0.2499,
         apr_cash=0.2999, foreign_txn_fee=0.03, rewards_rate=0.01, rewards_type="POINTS",
         min_credit_score=600, min_limit=4500, max_limit=12000, is_secured=False,
         is_business=False, share=0.14),
    dict(product_id="PRD_BUSINESS_REWARDS", product_name="Business Rewards",
         tier="BUSINESS", network="Visa", annual_fee=0.0, apr_purchase=0.2299,
         apr_cash=0.2899, foreign_txn_fee=0.0, rewards_rate=0.02, rewards_type="POINTS",
         min_credit_score=700, min_limit=8000, max_limit=45000, is_secured=False,
         is_business=True, share=0.12),
    dict(product_id="PRD_AMEX_GOLD", product_name="Gold Charge Rewards",
         tier="GOLD", network="American Express", annual_fee=250.0, apr_purchase=0.2099,
         apr_cash=0.2699, foreign_txn_fee=0.0, rewards_rate=0.03, rewards_type="POINTS",
         min_credit_score=740, min_limit=12000, max_limit=48000, is_secured=False,
         is_business=False, share=0.10),
]

# --------------------------------------------------------------------------
# Channels / entry modes
# --------------------------------------------------------------------------
CHANNELS = [
    dict(channel_id="CHN_POS_CHIP", channel_name="In Store - Chip", channel_group="IN_STORE",
         entry_mode="CHIP", is_card_present=True, base_share=0.30),
    dict(channel_id="CHN_POS_CONTACTLESS", channel_name="In Store - Contactless", channel_group="IN_STORE",
         entry_mode="CONTACTLESS", is_card_present=True, base_share=0.24),
    dict(channel_id="CHN_POS_SWIPE", channel_name="In Store - Swipe", channel_group="IN_STORE",
         entry_mode="SWIPE", is_card_present=True, base_share=0.03),
    dict(channel_id="CHN_ECOM", channel_name="E-Commerce", channel_group="ONLINE",
         entry_mode="ECOM", is_card_present=False, base_share=0.31),
    dict(channel_id="CHN_MOBILE_APP", channel_name="Mobile App", channel_group="MOBILE_APP",
         entry_mode="ECOM", is_card_present=False, base_share=0.09),
    dict(channel_id="CHN_MOTO", channel_name="Phone / Mail Order", channel_group="PHONE_ORDER",
         entry_mode="MOTO", is_card_present=False, base_share=0.02),
    dict(channel_id="CHN_ATM", channel_name="ATM Cash Advance", channel_group="ATM",
         entry_mode="CHIP", is_card_present=True, base_share=0.01),
]

# --------------------------------------------------------------------------
# MCC catalogue
# code, description, category, median USD, log-sigma, online share,
# weekend multiplier, base volume weight, foreign share
# --------------------------------------------------------------------------
MCC_CATALOG = [
    ("5411", "Grocery Stores, Supermarkets", "Grocery", 64, 0.55, 0.05, 1.15, 17.0, 0.01),
    ("5499", "Misc. Food Stores, Convenience", "Grocery", 13, 0.50, 0.08, 1.15, 4.0, 0.01),
    ("5812", "Eating Places, Restaurants", "Dining", 43, 0.70, 0.16, 1.30, 15.0, 0.02),
    ("5814", "Fast Food Restaurants", "Dining", 14, 0.50, 0.28, 1.20, 9.0, 0.01),
    ("5813", "Drinking Places (Bars)", "Dining", 49, 0.65, 0.05, 1.45, 2.0, 0.02),
    ("5541", "Service Stations (Fuel)", "Fuel", 52, 0.40, 0.02, 0.95, 7.0, 0.01),
    ("5310", "Discount Stores", "Retail", 58, 0.65, 0.10, 1.10, 6.0, 0.01),
    ("5311", "Department Stores", "Retail", 97, 0.70, 0.22, 1.10, 4.0, 0.01),
    ("5651", "Family Clothing Stores", "Apparel", 78, 0.65, 0.35, 1.05, 4.0, 0.02),
    ("5691", "Mens/Womens Clothing", "Apparel", 94, 0.65, 0.30, 1.05, 2.0, 0.02),
    ("5732", "Electronics Stores", "Electronics", 186, 0.90, 0.45, 1.00, 3.0, 0.04),
    ("5942", "Book Stores", "Retail", 28, 0.55, 0.55, 1.00, 1.0, 0.01),
    ("5945", "Toy and Hobby Shops", "Retail", 43, 0.60, 0.40, 1.20, 1.0, 0.01),
    ("5999", "Misc. Retail", "Retail", 46, 0.80, 0.50, 1.05, 3.0, 0.03),
    ("5399", "Misc. General Merchandise", "Retail", 55, 0.75, 0.35, 1.05, 2.0, 0.02),
    ("5200", "Home Supply Warehouse", "Home Improvement", 88, 0.75, 0.12, 1.25, 3.0, 0.01),
    ("5712", "Furniture Stores", "Home Furnishing", 320, 0.95, 0.22, 1.05, 1.0, 0.02),
    ("5722", "Household Appliance Stores", "Electronics", 480, 0.90, 0.28, 1.05, 1.0, 0.02),
    ("5912", "Drug Stores and Pharmacies", "Health", 26, 0.55, 0.14, 1.10, 6.0, 0.01),
    ("8099", "Health Practitioners", "Health", 146, 0.85, 0.10, 0.85, 2.0, 0.00),
    ("8021", "Dentists and Orthodontists", "Health", 212, 0.85, 0.05, 0.80, 1.0, 0.00),
    ("4511", "Airlines, Air Carriers", "Travel", 420, 0.80, 0.86, 1.00, 3.0, 0.18),
    ("7011", "Hotels, Motels, Resorts", "Travel", 265, 0.75, 0.72, 1.10, 3.0, 0.12),
    ("4722", "Travel Agencies", "Travel", 520, 0.95, 0.90, 1.00, 1.0, 0.15),
    ("4121", "Taxicabs and Rideshare", "Transport", 22, 0.50, 0.96, 1.20, 5.0, 0.05),
    ("4111", "Transit and Commuter Transport", "Transport", 12, 0.35, 0.45, 0.75, 2.0, 0.01),
    ("7512", "Automobile Rental Agency", "Travel", 240, 0.65, 0.60, 1.15, 1.0, 0.10),
    ("4900", "Utilities - Electric, Gas, Water", "Utilities", 138, 0.40, 0.25, 0.90, 3.0, 0.00),
    ("4814", "Telecommunication Services", "Telecom", 78, 0.30, 0.30, 0.95, 2.0, 0.00),
    ("4899", "Cable, Streaming and Pay TV", "Telecom", 62, 0.35, 0.62, 0.95, 2.0, 0.00),
    ("5815", "Digital Goods - Apps and Media", "Digital", 14, 0.65, 1.00, 1.00, 5.0, 0.02),
    ("5816", "Digital Games", "Digital", 23, 0.85, 1.00, 1.05, 2.0, 0.02),
    ("5968", "Subscription and Membership", "Digital", 19, 0.45, 1.00, 1.00, 3.0, 0.03),
    ("5964", "Direct Marketing - E-Commerce", "E-Commerce", 74, 0.90, 1.00, 1.00, 6.0, 0.06),
    ("5992", "Florists", "Retail", 56, 0.65, 0.70, 1.15, 1.0, 0.01),
    ("7832", "Motion Picture Theaters", "Entertainment", 33, 0.45, 0.30, 1.40, 1.0, 0.01),
    ("7995", "Gambling and Betting", "Entertainment", 62, 1.05, 0.80, 1.25, 1.0, 0.04),
    ("7997", "Fitness and Recreation Clubs", "Entertainment", 49, 0.45, 0.10, 1.00, 1.0, 0.00),
    ("7999", "Recreation Services", "Entertainment", 46, 0.75, 0.40, 1.20, 1.0, 0.01),
    ("5533", "Automotive Parts and Accessories", "Automotive", 98, 0.75, 0.30, 1.05, 1.0, 0.01),
    ("7538", "Automotive Service Shops", "Automotive", 286, 0.80, 0.05, 0.85, 1.0, 0.00),
    ("8999", "Professional Services", "Professional", 182, 1.00, 0.50, 0.90, 1.0, 0.01),
    ("8111", "Legal Services", "Professional", 420, 0.95, 0.20, 0.80, 1.0, 0.00),
    ("8220", "Colleges and Universities", "Education", 950, 0.95, 0.22, 0.70, 1.0, 0.01),
    ("8299", "Schools and Educational Services", "Education", 182, 0.85, 0.40, 0.80, 1.0, 0.00),
    ("5995", "Pet Shops and Supplies", "Pets", 56, 0.65, 0.35, 1.10, 2.0, 0.01),
    ("7349", "Building Cleaning and Maintenance", "Home Services", 120, 0.55, 0.10, 0.80, 1.0, 0.00),
]

# --------------------------------------------------------------------------
# US metros (top 25 by population) with sample cities and ZIP prefixes
# --------------------------------------------------------------------------
METROS = [
    dict(metro="New York-Newark-Jersey City, NY-NJ", cbsa="35620", region="Northeast", weight=8.0,
         cities=[("New York", "NY", "100", 40.7128, -74.0060), ("Brooklyn", "NY", "112", 40.6782, -73.9442),
                 ("Jersey City", "NJ", "073", 40.7178, -74.0431), ("Newark", "NJ", "071", 40.7357, -74.1724)]),
    dict(metro="Los Angeles-Long Beach-Anaheim, CA", cbsa="31080", region="West", weight=6.0,
         cities=[("Los Angeles", "CA", "900", 34.0522, -118.2437), ("Long Beach", "CA", "908", 33.7701, -118.1937),
                 ("Anaheim", "CA", "928", 33.8366, -117.9143), ("Pasadena", "CA", "911", 34.1478, -118.1445)]),
    dict(metro="Chicago-Naperville-Elgin, IL-IN", cbsa="16980", region="Midwest", weight=4.5,
         cities=[("Chicago", "IL", "606", 41.8781, -87.6298), ("Naperville", "IL", "605", 41.7508, -88.1535),
                 ("Evanston", "IL", "602", 42.0451, -87.6877)]),
    dict(metro="Dallas-Fort Worth-Arlington, TX", cbsa="19100", region="South", weight=4.0,
         cities=[("Dallas", "TX", "752", 32.7767, -96.7970), ("Fort Worth", "TX", "761", 32.7555, -97.3308),
                 ("Plano", "TX", "750", 33.0198, -96.6989), ("Arlington", "TX", "760", 32.7357, -97.1081)]),
    dict(metro="Houston-The Woodlands-Sugar Land, TX", cbsa="26420", region="South", weight=3.5,
         cities=[("Houston", "TX", "770", 29.7604, -95.3698), ("Sugar Land", "TX", "774", 29.6197, -95.6349),
                 ("The Woodlands", "TX", "773", 30.1658, -95.4613)]),
    dict(metro="Washington-Arlington-Alexandria, DC-VA-MD", cbsa="47900", region="South", weight=3.2,
         cities=[("Washington", "DC", "200", 38.9072, -77.0369), ("Arlington", "VA", "222", 38.8816, -77.0910),
                 ("Alexandria", "VA", "223", 38.8048, -77.0469), ("Bethesda", "MD", "208", 38.9847, -77.0947)]),
    dict(metro="Philadelphia-Camden-Wilmington, PA-NJ-DE", cbsa="37980", region="Northeast", weight=2.8,
         cities=[("Philadelphia", "PA", "191", 39.9526, -75.1652), ("Camden", "NJ", "081", 39.9259, -75.1196),
                 ("Wilmington", "DE", "198", 39.7391, -75.5398)]),
    dict(metro="Miami-Fort Lauderdale-West Palm Beach, FL", cbsa="33100", region="South", weight=2.8,
         cities=[("Miami", "FL", "331", 25.7617, -80.1918), ("Fort Lauderdale", "FL", "333", 26.1224, -80.1373),
                 ("West Palm Beach", "FL", "334", 26.7153, -80.0534), ("Hialeah", "FL", "330", 25.8576, -80.2781)]),
    dict(metro="Atlanta-Sandy Springs-Alpharetta, GA", cbsa="12060", region="South", weight=2.7,
         cities=[("Atlanta", "GA", "303", 33.7490, -84.3880), ("Sandy Springs", "GA", "303", 33.9304, -84.3733),
                 ("Alpharetta", "GA", "300", 34.0754, -84.2941)]),
    dict(metro="Phoenix-Mesa-Chandler, AZ", cbsa="38060", region="West", weight=2.5,
         cities=[("Phoenix", "AZ", "850", 33.4484, -112.0740), ("Mesa", "AZ", "852", 33.4152, -111.8315),
                 ("Chandler", "AZ", "852", 33.3062, -111.8413), ("Scottsdale", "AZ", "852", 33.4942, -111.9261)]),
    dict(metro="Boston-Cambridge-Newton, MA-NH", cbsa="14460", region="Northeast", weight=2.2,
         cities=[("Boston", "MA", "021", 42.3601, -71.0589), ("Cambridge", "MA", "021", 42.3736, -71.1097),
                 ("Newton", "MA", "024", 42.3370, -71.2092)]),
    dict(metro="San Francisco-Oakland-Berkeley, CA", cbsa="41860", region="West", weight=2.0,
         cities=[("San Francisco", "CA", "941", 37.7749, -122.4194), ("Oakland", "CA", "946", 37.8044, -122.2712),
                 ("Berkeley", "CA", "947", 37.8715, -122.2730)]),
    dict(metro="Riverside-San Bernardino-Ontario, CA", cbsa="40140", region="West", weight=2.0,
         cities=[("Riverside", "CA", "925", 33.9806, -117.3755), ("San Bernardino", "CA", "924", 34.1083, -117.2898),
                 ("Ontario", "CA", "917", 34.0633, -117.6509)]),
    dict(metro="Detroit-Warren-Dearborn, MI", cbsa="19820", region="Midwest", weight=1.9,
         cities=[("Detroit", "MI", "482", 42.3314, -83.0458), ("Warren", "MI", "480", 42.4775, -83.0277),
                 ("Dearborn", "MI", "481", 42.3223, -83.1763)]),
    dict(metro="Seattle-Tacoma-Bellevue, WA", cbsa="42660", region="West", weight=1.9,
         cities=[("Seattle", "WA", "981", 47.6062, -122.3321), ("Tacoma", "WA", "984", 47.2529, -122.4443),
                 ("Bellevue", "WA", "980", 47.6101, -122.2015)]),
    dict(metro="Minneapolis-St. Paul-Bloomington, MN-WI", cbsa="33460", region="Midwest", weight=1.7,
         cities=[("Minneapolis", "MN", "554", 44.9778, -93.2650), ("St. Paul", "MN", "551", 44.9537, -93.0900),
                 ("Bloomington", "MN", "554", 44.8408, -93.2983)]),
    dict(metro="San Diego-Chula Vista-Carlsbad, CA", cbsa="41740", region="West", weight=1.6,
         cities=[("San Diego", "CA", "921", 32.7157, -117.1611), ("Chula Vista", "CA", "919", 32.6401, -117.0842),
                 ("Carlsbad", "CA", "920", 33.1581, -117.3506)]),
    dict(metro="Tampa-St. Petersburg-Clearwater, FL", cbsa="45300", region="South", weight=1.5,
         cities=[("Tampa", "FL", "336", 27.9506, -82.4572), ("St. Petersburg", "FL", "337", 27.7676, -82.6403),
                 ("Clearwater", "FL", "337", 27.9659, -82.8001)]),
    dict(metro="Denver-Aurora-Lakewood, CO", cbsa="19740", region="West", weight=1.5,
         cities=[("Denver", "CO", "802", 39.7392, -104.9903), ("Aurora", "CO", "800", 39.7294, -104.8319),
                 ("Lakewood", "CO", "802", 39.7047, -105.0814)]),
    dict(metro="Baltimore-Columbia-Towson, MD", cbsa="12580", region="South", weight=1.3,
         cities=[("Baltimore", "MD", "212", 39.2904, -76.6122), ("Columbia", "MD", "210", 39.2037, -76.8610),
                 ("Towson", "MD", "212", 39.4015, -76.6019)]),
    dict(metro="St. Louis, MO-IL", cbsa="41180", region="Midwest", weight=1.2,
         cities=[("St. Louis", "MO", "631", 38.6270, -90.1994), ("Clayton", "MO", "631", 38.6426, -90.3237)]),
    dict(metro="Charlotte-Concord-Gastonia, NC-SC", cbsa="16740", region="South", weight=1.2,
         cities=[("Charlotte", "NC", "282", 35.2271, -80.8431), ("Concord", "NC", "280", 35.4088, -80.5795),
                 ("Gastonia", "NC", "280", 35.2621, -81.1873)]),
    dict(metro="Orlando-Kissimmee-Sanford, FL", cbsa="36740", region="South", weight=1.2,
         cities=[("Orlando", "FL", "328", 28.5383, -81.3792), ("Kissimmee", "FL", "347", 28.2920, -81.4076),
                 ("Sanford", "FL", "327", 28.8006, -81.2731)]),
    dict(metro="San Antonio-New Braunfels, TX", cbsa="41700", region="South", weight=1.1,
         cities=[("San Antonio", "TX", "782", 29.4241, -98.4936), ("New Braunfels", "TX", "781", 29.7030, -98.1245)]),
    dict(metro="Portland-Vancouver-Hillsboro, OR-WA", cbsa="38900", region="West", weight=1.1,
         cities=[("Portland", "OR", "972", 45.5152, -122.6784), ("Vancouver", "WA", "986", 45.6387, -122.6615),
                 ("Hillsboro", "OR", "971", 45.5229, -122.9898)]),
]

STATE_TZ = {
    "NY": "America/New_York", "NJ": "America/New_York", "DC": "America/New_York",
    "VA": "America/New_York", "MD": "America/New_York", "PA": "America/New_York",
    "DE": "America/New_York", "MA": "America/New_York", "NH": "America/New_York",
    "GA": "America/New_York", "NC": "America/New_York", "SC": "America/New_York",
    "FL": "America/New_York", "OH": "America/New_York", "MI": "America/Detroit",
    "IL": "America/Chicago", "IN": "America/Indiana/Indianapolis", "TX": "America/Chicago",
    "MN": "America/Chicago", "MO": "America/Chicago", "WI": "America/Chicago",
    "CO": "America/Denver", "AZ": "America/Phoenix", "CA": "America/Los_Angeles",
    "WA": "America/Los_Angeles", "OR": "America/Los_Angeles",
}

# International merchant locations used for cross-border spend
INTL_MERCHANTS_GEO = [
    ("Toronto", "ON", "CA", "M5V", "America/Toronto", 43.6532, -79.3832),
    ("Vancouver", "BC", "CA", "V6B", "America/Vancouver", 49.2827, -123.1207),
    ("London", "ENG", "GB", "EC1A", "Europe/London", 51.5074, -0.1278),
    ("Paris", "IDF", "FR", "75001", "Europe/Paris", 48.8566, 2.3522),
    ("Mexico City", "CMX", "MX", "06000", "America/Mexico_City", 19.4326, -99.1332),
    ("Tokyo", "TKY", "JP", "100-0001", "Asia/Tokyo", 35.6762, 139.6503),
    ("Sydney", "NSW", "AU", "-2000", "Australia/Sydney", -33.8688, 151.2093),
    ("Madrid", "MAD", "ES", "28001", "Europe/Madrid", 40.4168, -3.7038),
]

# --------------------------------------------------------------------------
# Merchant brand pools by category
# --------------------------------------------------------------------------
MERCHANT_BRANDS = {
    "Grocery": ["FreshLine Market", "Harvest Grocers", "GreenCart Foods", "Everyday Market",
                "Copperfield Grocery", "Northgate Foods", "Sunrise Market", "Basket & Barrel"],
    "Dining": ["Urban Table", "The Copper Fork", "Riverside Grill", "Basil & Stone",
               "Northside Kitchen", "Ember Steakhouse", "Saffron House", "Blue Harbor Bistro",
               "Taco Loco", "Pho District"],
    "Fuel": ["Northstar Fuel", "Eagle Point Gas", "Metro Fuel Stop", "Summit Energy"],
    "Retail": ["Evergreen Goods", "The Supply Co", "Meridian Home", "Parkside Living",
               "Hearth & Home", "Bluebird Outfitters", "Lakeshore Living"],
    "Apparel": ["Thread & Co", "Nova Apparel", "Rue Meridian", "Coastline Denim",
                "Willow & Vine", "Standard Issue"],
    "Electronics": ["Volt Electronics", "Circuit House", "ByteWorks", "Harbor Tech"],
    "Health": ["CareFirst Pharmacy", "Wellpoint Drug", "HealthMart Express", "MediSave"],
    "Travel": ["Skyway Airlines", "Altura Air", "Harbor Hotels", "Summit Lodge",
               "Voyager Travel", "Meridian Resorts"],
    "Transport": ["CityRide", "Metro Hail", "TransitLink"],
    "Utilities": ["Metro Power & Light", "Clearwater Utilities", "Lakeside Gas Co"],
    "Telecom": ["Nexa Wireless", "FiberLine", "SignalOne"],
    "Digital": ["CloudStream", "PixelPlay", "SoundWave", "Nova Apps"],
    "E-Commerce": ["ShopNest", "Cartwheel", "PrimeBasket", "DirectBuy Online"],
    "Entertainment": ["Starlight Cinemas", "Playhouse Live", "Ace Sports Club"],
    "Automotive": ["AutoZone Express", "GearWorks", "PitStop Service"],
    "Professional": ["Sterling Consulting", "Harbor Legal", "Crestline Advisory"],
    "Education": ["Ridgeline University", "Lakeview Academy", "Summit Learning"],
    "Pets": ["PawMart", "Happy Tails Supply"],
    "Home Improvement": ["BuildRight", "Timber & Tools"],
    "Home Furnishing": ["Casa Moderna", "RoomCraft"],
    "Home Services": ["SparkleHome", "ClearView Cleaning"],
}

# --------------------------------------------------------------------------
# People
# --------------------------------------------------------------------------
FIRST_NAMES_M = [
    "James","Michael","Robert","David","William","Richard","Joseph","Thomas","Christopher","Charles",
    "Daniel","Matthew","Anthony","Mark","Donald","Steven","Paul","Andrew","Joshua","Kenneth",
    "Kevin","Brian","George","Timothy","Ronald","Edward","Jason","Jeffrey","Ryan","Jacob",
    "Gary","Nicholas","Eric","Jonathan","Stephen","Larry","Justin","Scott","Brandon","Benjamin",
    "Samuel","Gregory","Alexander","Patrick","Frank","Raymond","Jack","Dennis","Jerry","Tyler",
    "Aaron","Jose","Adam","Nathan","Henry","Douglas","Zachary","Peter","Kyle","Walter",
    "Ethan","Jeremy","Harold","Keith","Christian","Roger","Noah","Gerald","Carl","Terry",
    "Sean","Austin","Arthur","Lawrence","Jesse","Dylan","Bryan","Joe","Jordan","Billy",
    "Bruce","Albert","Willie","Gabriel","Logan","Alan","Juan","Wayne","Roy","Ralph",
]
FIRST_NAMES_F = [
    "Mary","Patricia","Jennifer","Linda","Elizabeth","Barbara","Susan","Jessica","Sarah","Karen",
    "Lisa","Nancy","Betty","Margaret","Sandra","Ashley","Kimberly","Emily","Donna","Michelle",
    "Carol","Amanda","Dorothy","Melissa","Deborah","Stephanie","Rebecca","Sharon","Laura","Cynthia",
    "Kathleen","Amy","Angela","Shirley","Anna","Brenda","Pamela","Emma","Nicole","Helen",
    "Samantha","Katherine","Christine","Debra","Rachel","Carolyn","Janet","Catherine","Maria","Heather",
    "Diane","Ruth","Julie","Olivia","Joyce","Virginia","Victoria","Kelly","Lauren","Christina",
    "Joan","Evelyn","Judith","Megan","Andrea","Cheryl","Hannah","Jacqueline","Martha","Gloria",
    "Teresa","Ann","Sara","Madison","Frances","Kathryn","Janice","Jean","Abigail","Alice",
    "Julia","Judy","Sophia","Grace","Denise","Amber","Doris","Marilyn","Danielle","Beverly",
]
SURNAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
    "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
    "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson",
    "Walker","Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
    "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts",
    "Gomez","Phillips","Evans","Turner","Diaz","Parker","Cruz","Edwards","Collins","Reyes",
    "Stewart","Morris","Morales","Murphy","Cook","Rogers","Gutierrez","Ortiz","Morgan","Cooper",
    "Peterson","Bailey","Reed","Kelly","Howard","Ramos","Kim","Cox","Ward","Richardson",
    "Watson","Brooks","Chavez","Wood","James","Bennett","Gray","Mendoza","Ruiz","Hughes",
    "Price","Alvarez","Castillo","Sanders","Patel","Myers","Long","Ross","Foster","Jimenez",
    "Powell","Jenkins","Perry","Russell","Sullivan","Bell","Coleman","Butler","Henderson","Barnes",
    "Gonzales","Fisher","Vasquez","Simmons","Romero","Jordan","Patterson","Alexander","Hamilton","Graham",
]
OCCUPATIONS = [
    ("Software Engineer", "Information Technology", 128000),
    ("Registered Nurse", "Healthcare", 86000),
    ("Teacher", "Education", 64000),
    ("Accountant", "Finance", 78000),
    ("Sales Associate", "Retail", 38000),
    ("Project Manager", "Professional Services", 98000),
    ("Truck Driver", "Transportation", 56000),
    ("Electrician", "Construction", 68000),
    ("Marketing Manager", "Marketing", 92000),
    ("Customer Service Rep", "Business Services", 42000),
    ("Physician", "Healthcare", 235000),
    ("Attorney", "Legal", 145000),
    ("Dental Hygienist", "Healthcare", 81000),
    ("Mechanical Engineer", "Manufacturing", 96000),
    ("Graphic Designer", "Creative", 62000),
    ("Restaurant Manager", "Hospitality", 58000),
    ("Financial Analyst", "Finance", 88000),
    ("Data Analyst", "Information Technology", 84000),
    ("Police Officer", "Public Sector", 72000),
    ("Warehouse Associate", "Logistics", 39000),
    ("Pharmacist", "Healthcare", 132000),
    ("Small Business Owner", "Business Services", 110000),
    ("Retired", "Retired", 44000),
    ("Student", "Student", 14000),
    ("Self-Employed Consultant", "Professional Services", 105000),
    ("Administrative Assistant", "Business Services", 46000),
    ("Chef", "Hospitality", 54000),
    ("Civil Engineer", "Engineering", 90000),
    ("Physical Therapist", "Healthcare", 95000),
    ("Real Estate Agent", "Real Estate", 76000),
]
EMPLOYERS = [
    "Meridian Health System", "Northstar Technologies", "Blue Ridge Logistics", "Summit Financial Group",
    "Crestline Manufacturing", "Harbor Point Schools", "Evergreen Retail Group", "Metro Transit Authority",
    "Lakeside Medical Center", "Pinnacle Consulting", "Vertex Solutions", "Copperfield Foods",
    "Brightline Media", "Anchor Insurance", "Redwood Analytics", "Sterling Hospitality",
    "Self-Employed", "City of Phoenix", "State University System", "United Distribution Co",
]

US_BANKS = [
    ("JPMORGAN CHASE", "CHASUS33XXX", "021000021"),
    ("BANK OF AMERICA", "BOFAUS3NXXX", "026009593"),
    ("WELLS FARGO", "WFBIUS6SXXX", "121000248"),
    ("CITIBANK", "CITIUS33XXX", "021000089"),
    ("US BANK", "USBKUS44XXX", "091000022"),
    ("PNC BANK", "PNCCUS33XXX", "043000096"),
    ("CAPITAL ONE", "NFBKUS33XXX", "051405515"),
    ("TD BANK", "NRTHUS33XXX", "031101266"),
    ("ALLY BANK", "ALYIUS33XXX", "124003116"),
    ("SCHWAB BANK", "SCHBUS33XXX", "121202211"),
]

US_CITIES_BIRTH = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
    "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA",
    "Austin, TX", "Jacksonville, FL", "Fort Worth, TX", "Columbus, OH", "Charlotte, NC",
    "Indianapolis, IN", "San Francisco, CA", "Seattle, WA", "Denver, CO", "Nashville, TN",
    "Oklahoma City, OK", "El Paso, TX", "Boston, MA", "Portland, OR", "Las Vegas, NV",
    "Detroit, MI", "Memphis, TN", "Louisville, KY", "Baltimore, MD", "Milwaukee, WI",
    "Albuquerque, NM", "Tucson, AZ", "Fresno, CA", "Sacramento, CA", "Kansas City, MO",
    "Mesa, AZ", "Atlanta, GA", "Omaha, NE", "Colorado Springs, CO", "Raleigh, NC",
    "Miami, FL", "Long Beach, CA", "Virginia Beach, VA", "Oakland, CA", "Minneapolis, MN",
    "Tulsa, OK", "Tampa, FL", "Arlington, TX", "New Orleans, LA", "Wichita, KS",
    "Bakersfield, CA", "Aurora, CO", "Anaheim, CA", "Santa Ana, CA", "St. Louis, MO",
    "Riverside, CA", "Corpus Christi, TX", "Pittsburgh, PA", "Lexington, KY", "Anchorage, AK",
]

ACQUISITION_CHANNELS = ["Web Search", "Paid Social", "Direct Mail", "Partner Referral",
                        "In-App Offer", "Affiliate", "Branch", "Comparison Site"]
ID_TYPES = ["DRIVER_LICENSE", "STATE_ID", "PASSPORT", "ITIN", "MILITARY_ID"]
EMPLOYMENT_STATUS = ["EMPLOYED", "SELF_EMPLOYED", "RETIRED", "STUDENT", "UNEMPLOYED"]
HOUSING_STATUS = ["MORTGAGE", "RENT", "OWN", "LIVE_WITH_FAMILY"]
EDUCATION = ["HIGH_SCHOOL", "SOME_COLLEGE", "ASSOCIATE", "BACHELOR", "GRADUATE", "DOCTORATE"]
MARITAL_STATUS = ["SINGLE", "MARRIED", "DIVORCED", "WIDOWED", "DOMESTIC_PARTNER"]

DECLINE_REASONS = {
    "CREDIT_LIMIT_EXCEEDED": "Insufficient available credit",
    "SUSPECTED_FRAUD": "Blocked by fraud rules - step-up required",
    "INVALID_CARD": "Invalid card number or expiry",
    "DO_NOT_HONOR": "Issuer declined - do not honor",
    "CARD_LOCKED": "Card is locked by cardholder",
    "EXPIRED_CARD": "Card expired",
    "INCORRECT_PIN": "Incorrect PIN entered",
    "MERCHANT_BLOCKED": "Merchant category blocked",
}

FRAUD_TYPES = {
    "CARD_NOT_PRESENT": "Card-not-present fraud",
    "ACCOUNT_TAKEOVER": "Account takeover",
    "COUNTERFEIT": "Counterfeit card",
    "LOST_STOLEN": "Lost or stolen card",
    "FRIENDLY_FRAUD": "First-party (friendly) fraud",
    "APPLICATION_FRAUD": "Application fraud",
}

DISPUTE_OUTCOMES = ["FRAUD_CONFIRMED", "CUSTOMER_ERROR", "MERCHANT_CREDIT", "INCONCLUSIVE", "WITHDRAWN"]

# --------------------------------------------------------------------------
# Digital event catalogue (mirrors the source BNPL app journey)
# --------------------------------------------------------------------------
APP_EVENTS = [
    ("app_launch", 0.20),
    ("card_home_viewed", 0.18),
    ("card_activation_intro_viewed", 0.05),
    ("card_activation_button_clicked", 0.04),
    ("kyc_document_viewed", 0.03),
    ("kyc_permission_viewed", 0.03),
    ("application_form_viewed", 0.03),
    ("application_submitted", 0.02),
    ("credit_decision_viewed", 0.02),
    ("card_activated_dashboard_viewed", 0.07),
    ("card_details_viewed", 0.04),
    ("statement_list_viewed", 0.06),
    ("statement_detail_viewed", 0.05),
    ("payment_page_viewed", 0.05),
    ("payment_made", 0.04),
    ("payment_failed", 0.01),
    ("installment_plan_viewed", 0.02),
    ("installment_plan_selected", 0.01),
    ("rewards_page_viewed", 0.03),
    ("card_lock_toggled", 0.01),
    ("fraud_alert_viewed", 0.01),
    ("dispute_started", 0.005),
    ("support_chat_opened", 0.02),
    ("notification_opened", 0.04),
    ("settings_viewed", 0.03),
]

ANDROID_DEVICES = ["samsung SM-A135F", "samsung SM-S911B", "Google Pixel 7", "Google Pixel 8a",
                   "motorola moto g power", "Xiaomi 2107113SG", "OnePlus CPH2451", "HONOR BRP-NX1"]
IOS_DEVICES = ["Apple iPhone17,2", "Apple iPhone14,8", "Apple iPhone15,3", "Apple iPhone13,2", "Apple iPhone16,1"]
WEB_BROWSERS = ["Chrome 141 / Windows", "Safari 18 / macOS", "Edge 141 / Windows",
                "Chrome 141 / Android", "Safari 18 / iOS"]


# --------------------------------------------------------------------------
# Ticket calibration (computed once, shared by facts.py and txgen.py)
# --------------------------------------------------------------------------
def _calibrate_amounts():
    """Monotone transform a * x^b so the MCC-weighted ticket mixture matches the
    US target median and p95 while preserving category ordering."""
    rs = np.random.default_rng(90210)
    codes = [m[0] for m in MCC_CATALOG]
    med = np.array([m[3] for m in MCC_CATALOG], float)
    sig = np.array([m[4] for m in MCC_CATALOG], float)
    w = np.array([m[7] for m in MCC_CATALOG], float)
    w = w / w.sum()
    n = 120_000
    idx = rs.choice(len(codes), size=n, p=w)
    vals = np.clip(rs.lognormal(np.log(med[idx]), sig[idx] * 0.75), 1.0, 20_000.0)
    m0 = float(np.median(vals))
    p95 = float(np.quantile(vals, 0.95))
    b = float(np.log(US_TICKET_P95 / US_TICKET_MEDIAN) / np.log(p95 / m0))
    a = float(US_TICKET_MEDIAN / (m0 ** b))
    return a, b


AMOUNT_A, AMOUNT_B = _calibrate_amounts()

# --------------------------------------------------------------------------
# Delinquency lifecycle (reserved for a later build)
# --------------------------------------------------------------------------
# See docs/delinquency_extension.md. These values are inert until
# DELINQUENCY_LIFECYCLE_ENABLED is True and the transition block is added to the
# statement-cycle loop in build/facts.py.
DELINQUENCY_LIFECYCLE_ENABLED = False
DEFAULT_DPD = 180
CURE_WINDOW_DAYS = 30
RECOVERY_TARGET_PCT = (0.15, 0.25)

# --------------------------------------------------------------------------
# v7 charge-off incidence calibration
# --------------------------------------------------------------------------
# v6 produced only 4 charge-offs, which makes any recovery/collection
# distributional claim meaningless. A small share of accounts is placed on a
# chronic non-payment path from their first statement so that DPD crosses 180
# inside the 13-month window. The rate is chosen to yield well above the 400
# charge-off release gate without changing the overall delinquency mix beyond
# the documented bands.
CHARGE_OFF_DISTRESS_SHARE = 0.025
CHARGE_OFF_MIN_COUNT = 400

# --------------------------------------------------------------------------
# v7 ML extract settings
# --------------------------------------------------------------------------
ML_DIR_NAME = "ml"
ML_FRAUD_LABEL_NOISE_INDISTINGUISHABLE = 0.15   # true fraud relabelled 0
ML_FRAUD_LABEL_NOISE_FALSE_FLAG = 0.02          # non-fraud relabelled 1
ML_OBSERVATION_DATE = dt.date(2026, 3, 31)      # credit/churn feature cut-off
ML_OUTCOME_END = dt.date(2026, 9, 30)           # label outcome window end
ML_FRAUD_OOT_START = dt.date(2026, 7, 1)        # final-quarter out-of-time test
ML_FORECAST_TEST_START = dt.date(2026, 7, 1)      # forecast test window; seasonality/merchants use earlier months only
ML_FRAUD_MAX_ROWS = 2_000_000                   # sampled non-fraud rows
ML_FORECAST_MERCHANTS = 5_000                   # largest merchants by spend
ML_PR_AUC_MAX = 0.95                            # realistic, not separable
ML_DRIFT_PSI_MIN = 0.20                         # final-quarter drift must fire
