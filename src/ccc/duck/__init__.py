"""Open the split DuckDB warehouse through its hub database."""

from pathlib import Path

import duckdb

DATABASES = ("dim", "facts", "gold")


def _sql_string(value: str | Path) -> str:
    return str(value).replace("'", "''")


def connect(out_dir: str | Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open hub.duckdb and attach the warehouse databases by their stable aliases."""
    output_root = Path(out_dir).expanduser().resolve()
    current = output_root / "current"
    try:
        root = current.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"warehouse current link is missing: {current}") from exc
    hub = root / "hub.duckdb"
    if not hub.is_file():
        raise FileNotFoundError(f"warehouse hub is missing: {hub}")
    con = duckdb.connect(str(hub), read_only=read_only)
    try:
        con.execute("SET TimeZone = 'UTC'")
        for alias in DATABASES:
            mode = " (READ_ONLY)" if read_only else ""
            con.execute(f"ATTACH '{_sql_string(root / f'{alias}.duckdb')}' AS {alias}{mode}")
        return con
    except BaseException:
        con.close()
        raise


__all__ = ["connect"]
