"""Post-processing fixes applied after the simulator.

Keeps realism details that are cheaper to enforce on the materialised tables:

* declined / reversed authorisations carry no auth code and no decline reason
  is attached to approved ones;
* `is_recurring` is driven by the merchant rather than sampled per transaction;
* cards attached to closed or charged-off accounts are cancelled.

The transaction fact is large (hundreds of MB on disk), so it is processed with
Polars lazily and streamed back to parquet instead of being loaded into pandas.

Invoked by `build.run_all` before final validation. For manual repairs only,
run `python -m build.postprocess`, then `python -m build.refresh_docs`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl

from . import config as C
from .utils import log, parquet_metadata


def hygiene(tx: pl.LazyFrame, merch: pl.DataFrame, acct: pl.DataFrame) -> pl.LazyFrame:
    """Transaction hygiene rules; row-for-row equivalent of the former pandas logic."""
    approved = (pl.col("status") == "APPROVED").fill_null(False)

    recurring = (
        merch.lazy()
        .select("merchant_id", pl.col("is_recurring").alias("_recurring"))
        .unique(subset="merchant_id", keep="first", maintain_order=True)
    )
    network = (
        acct.lazy()
        .select("account_id", pl.col("network").alias("_network"))
        .unique(subset="account_id", keep="first", maintain_order=True)
    )

    dt_dtype = tx.collect_schema()["transaction_date"]
    if dt_dtype.base_type() is pl.Datetime:
        tdate = pl.col("transaction_date")
    else:  # date / string -> datetime, as pd.to_datetime did
        tdate = pl.col("transaction_date").cast(pl.Datetime("ns"))

    return (
        tx.with_columns(
            pl.when(approved).then(pl.col("auth_code")).otherwise(None).alias("auth_code"),
            pl.when(approved).then(None).otherwise(pl.col("decline_reason_code"))
            .alias("decline_reason_code"),
            pl.when(approved).then(None).otherwise(pl.col("decline_reason"))
            .alias("decline_reason"),
            tdate.alias("transaction_date"),
        )
        .join(recurring, on="merchant_id", how="left", maintain_order="left")
        .join(network, on="account_id", how="left", maintain_order="left")
        .with_columns(
            pl.col("_recurring").fill_null(False).cast(pl.Boolean).alias("is_recurring"),
            pl.col("_network").alias("network"),
        )
        .drop("_recurring", "_network")
        .sort(["transaction_date", "transaction_ts"], maintain_order=True, nulls_last=True)
    )


def cancel_cards(cards: pl.DataFrame, acct: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Cancel ACTIVE cards whose account is CLOSED / CHARGED_OFF."""
    dead = (
        acct.filter(pl.col("account_status").is_in(["CLOSED", "CHARGED_OFF"]))
        .get_column("account_id").drop_nulls().unique()
    )
    mask = pl.col("account_id").is_in(dead.implode()).fill_null(False) & (
        pl.col("card_status") == "ACTIVE"
    ).fill_null(False)
    n = cards.select(mask.sum()).item()
    out = cards.with_columns(
        pl.when(mask).then(pl.lit("CANCELLED")).otherwise(pl.col("card_status"))
        .alias("card_status")
    )
    return out, int(n)


def _sink(lf: pl.LazyFrame, path: Path, table_name: str) -> None:
    """Stream to a temp file then atomically replace (the source may be `path`)."""
    tmp = path.with_name(path.name + ".tmp")
    md = parquet_metadata(table_name, path)
    # the arrow schema carries the metadata pyarrow readers see as schema.metadata
    schema = pl.DataFrame(schema=lf.collect_schema()).to_arrow().schema.with_metadata(md)
    lf.sink_parquet(tmp, compression="snappy", metadata=md, arrow_schema=schema)
    os.replace(tmp, path)


def main() -> int:
    tx_path = C.FACT_DIR / "fact_transaction.parquet"
    merch = pl.read_parquet(C.DIM_DIR / "dim_merchant.parquet", columns=["merchant_id", "is_recurring"])
    acct = pl.read_parquet(C.DIM_DIR / "dim_account.parquet")
    cards = pl.read_parquet(C.DIM_DIR / "dim_card.parquet")

    # --- transaction hygiene -------------------------------------------------
    _sink(hygiene(pl.scan_parquet(tx_path), merch, acct), tx_path, "fact_transaction")
    n = pl.scan_parquet(tx_path).select(pl.len()).collect().item()
    log(f"transaction hygiene applied to {n:,} rows")

    # --- card lifecycle ------------------------------------------------------
    cards, cancelled = cancel_cards(cards, acct)
    card_path = C.DIM_DIR / "dim_card.parquet"
    _sink(cards.lazy(), card_path, "dim_card")
    log(f"cancelled {cancelled} cards on closed accounts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
