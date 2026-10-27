"""Open the split DuckDB warehouse through its hub database."""

from pathlib import Path

import duckdb

DATABASES = ("dim", "facts", "gold")


def _sql_string(value: str | Path) -> str:
    return str(value).replace("'", "''")


def connect(out_dir: str | Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open hub.duckdb and attach the warehouse databases by their stable aliases."""
    root = Path(out_dir).expanduser().resolve()
    con = duckdb.connect(str(root / "hub.duckdb"), read_only=read_only)
    try:
        for alias in DATABASES:
            mode = " (READ_ONLY)" if read_only else ""
            con.execute(f"ATTACH '{_sql_string(root / f'{alias}.duckdb')}' AS {alias}{mode}")
        return con
    except BaseException:
        con.close()
        raise


__all__ = ["connect"]
