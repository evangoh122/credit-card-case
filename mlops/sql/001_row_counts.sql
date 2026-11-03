-- Versioned MLOps check: row counts across dim, facts and gold.
-- Run with: uv run python -m ccc.mlops.sql_runner --db <warehouse> --sql mlops/sql/001_row_counts.sql
SELECT 'dim.dim_customer' AS table_name, count(*) AS row_count FROM dim.dim_customer
UNION ALL
SELECT 'facts.fact_transactions' AS table_name, count(*) AS row_count FROM facts.fact_transactions
UNION ALL
SELECT 'facts.fact_events' AS table_name, count(*) AS row_count FROM facts.fact_events
UNION ALL
SELECT 'gold.fact_rfm_snapshot' AS table_name, count(*) AS row_count FROM gold.fact_rfm_snapshot
UNION ALL
SELECT 'gold.fact_engagement_snapshot' AS table_name, count(*) AS row_count
FROM gold.fact_engagement_snapshot
ORDER BY table_name;
