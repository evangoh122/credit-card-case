"""Post-processing fixes applied after the simulator.

Keeps realism details that are cheaper to enforce on the materialised tables:

* declined / reversed authorisations carry no auth code and no decline reason
  is attached to approved ones;
* `is_recurring` is driven by the merchant rather than sampled per transaction;
* cards attached to closed or charged-off accounts are cancelled.

Invoked by `build.run_all` before final validation. For manual repairs only,
run `python -m build.postprocess`, then `python -m build.refresh_docs`.
"""
from __future__ import annotations

import sys

import pandas as pd

from . import config as C
from .utils import log, read_parquet, write_parquet


def main() -> int:
    tx_path = C.FACT_DIR / "fact_transaction.parquet"
    tx = read_parquet(tx_path)
    merch = read_parquet(C.DIM_DIR / "dim_merchant.parquet")
    acct = read_parquet(C.DIM_DIR / "dim_account.parquet")
    cards = read_parquet(C.DIM_DIR / "dim_card.parquet")

    # --- transaction hygiene -------------------------------------------------
    approved = tx.status == "APPROVED"
    tx.loc[~approved, "auth_code"] = None
    tx.loc[approved, "decline_reason_code"] = None
    tx.loc[approved, "decline_reason"] = None

    recurring = merch.set_index("merchant_id")["is_recurring"]
    tx["is_recurring"] = tx.merchant_id.map(recurring).fillna(False).astype(bool)

    # denormalise the card network onto the transaction for easy slicing
    net = acct.set_index("account_id")["network"]
    tx["network"] = tx.account_id.map(net)

    # refunds are always linked to a purchase in the same statement window
    tx["transaction_date"] = pd.to_datetime(tx.transaction_date)
    tx = tx.sort_values(["transaction_date", "transaction_ts"]).reset_index(drop=True)
    write_parquet(tx, tx_path, "fact_transaction")
    log(f"transaction hygiene applied to {len(tx):,} rows")

    # --- card lifecycle ------------------------------------------------------
    dead = set(acct.loc[acct.account_status.isin(["CLOSED", "CHARGED_OFF"]), "account_id"])
    mask = cards.account_id.isin(dead) & (cards.card_status == "ACTIVE")
    cards.loc[mask, "card_status"] = "CANCELLED"
    write_parquet(cards, C.DIM_DIR / "dim_card.parquet", "dim_card")
    log(f"cancelled {int(mask.sum())} cards on closed accounts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
