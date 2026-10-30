"""Regenerate documentation from the materialised parquet files.

Useful after a manual fix to a table, or to re-run validation without
re-simulating the whole book.

Usage:
    python -m build.refresh_docs
"""
from __future__ import annotations

import sys

from . import config as C
from . import dictionary, validation
from .utils import log, read_parquet
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

# fact_transaction is the one table too large to hold in pandas: it stays a lazy
# Polars scan and validation / dictionary / inventory profile it from the file.
LAZY_FACTS = ("fact_transaction",)


def load_all() -> tuple[dict, dict, dict]:
    dims, facts, raw = {}, {}, {}
    for p in sorted(C.DIM_DIR.glob("*.parquet")):
        dims[p.stem] = read_parquet(p)
    for p in sorted(C.FACT_DIR.glob("*.parquet")):
        facts[p.stem] = pl.scan_parquet(p) if p.stem in LAZY_FACTS else read_parquet(p)
    for p in sorted(C.RAW_DIR.glob("*.parquet")):
        raw[p.stem] = read_parquet(p)
    return dims, facts, raw


def main() -> int:
    dims, facts, raw = load_all()
    log(f"loaded {len(dims)} dims, {len(facts)} facts, {len(raw)} raw tables")

    report = validation.run(dims, facts, raw)
    (C.DOC_DIR / "validation_report.md").write_text(report, encoding="utf-8")
    failed = validation.LAST_FAIL_COUNT

    all_tables = {**dims, **facts, **raw}
    dd, dd_md = dictionary.build(all_tables)
    dd.to_csv(C.DOC_DIR / "data_dictionary.csv", index=False, encoding="utf-8")
    (C.DOC_DIR / "data_dictionary.md").write_text(dd_md, encoding="utf-8")

    # refresh the physical inventory too (postprocess rewrites fact_transaction)
    inv = []
    for sub in ("dim", "fact", "raw"):
        for p in sorted((C.OUT / sub).glob("*.parquet")):
            inv.append(dict(table=p.stem, rows=pq.ParquetFile(p).metadata.num_rows,
                            mb=round(p.stat().st_size / 1024 / 1024, 2)))
    pd.DataFrame(inv).to_csv(C.DOC_DIR / "table_inventory.csv", index=False)
    log("docs refreshed" + (f" -- {failed} FAILURES" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
