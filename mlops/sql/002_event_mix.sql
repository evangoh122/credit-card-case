-- Versioned MLOps check: event-type mix in the facts database.
SELECT event_type, count(*) AS event_count
FROM facts.fact_events
GROUP BY event_type
ORDER BY event_type;
