"""SQL layer (DuckDB): player metrics for the readout, and events aggregated back to player-days."""
from __future__ import annotations

import duckdb
import pandas as pd

PLAYER_METRICS = """
SELECT
    player_id,
    SUM(CASE WHEN day >= 0 THEN ad_rev + iap_rev END)                  AS revenue,
    SUM(CASE WHEN day >= 0 THEN ad_rev END)                            AS ad_rev,
    SUM(CASE WHEN day >= 0 THEN iap_rev END)                           AS iap_rev,
    MAX(CASE WHEN day BETWEEN 7 AND 13 AND active THEN 1 ELSE 0 END)   AS returned_week2,
    SUM(CASE WHEN day >= 0 AND active THEN 1 ELSE 0 END)               AS active_days,
    SUM(CASE WHEN day >= 0 THEN sessions END)                          AS sessions,
    SUM(CASE WHEN day >= 0 THEN completes END)                         AS levels_completed,
    -- pre-experiment covariates for CUPED (days -14 to -1, before anyone saw the feature)
    SUM(CASE WHEN day < 0 THEN ad_rev + iap_rev END)                   AS pre_revenue,
    SUM(CASE WHEN day < 0 THEN ad_rev END)                             AS pre_ad_rev,
    SUM(CASE WHEN day < 0 THEN iap_rev END)                            AS pre_iap_rev,
    MAX(CASE WHEN day BETWEEN -7 AND -1 AND active THEN 1 ELSE 0 END)  AS pre_returned_week,
    SUM(CASE WHEN day < 0 AND active THEN 1 ELSE 0 END)                AS pre_active_days,
    SUM(CASE WHEN day < 0 THEN sessions END)                           AS pre_sessions,
    SUM(CASE WHEN day < 0 THEN completes END)                          AS pre_levels_completed,
    SUM(CASE WHEN day < 0 THEN rewarded END)                           AS pre_rewarded
FROM daily
GROUP BY player_id
ORDER BY player_id
"""

EVENTS_TO_DAYS = """
WITH s AS (
    SELECT session_id, player_id, FLOOR((ts - {t0}) / 86400)::INT AS day
    FROM events WHERE event = 'session_start'
)
SELECT
    s.player_id, s.day,
    COUNT(DISTINCT s.session_id)                                                 AS sessions,
    COUNT(*) FILTER (WHERE e.event = 'level_end')                                AS attempts,
    COUNT(*) FILTER (WHERE e.event = 'level_end' AND e.result = 'complete')      AS completes,
    COUNT(*) FILTER (WHERE e.event = 'continue_used')                            AS continues,
    COUNT(*) FILTER (WHERE e.event = 'ad_impression' AND e.format = 'interstitial') AS interstitials,
    COUNT(*) FILTER (WHERE e.event = 'ad_impression' AND e.format = 'rewarded')  AS rewarded,
    ROUND(COALESCE(SUM(e.revenue_usd) FILTER (WHERE e.event = 'ad_impression'), 0), 6) AS ad_rev,
    COUNT(*) FILTER (WHERE e.event = 'iap_purchase')                             AS iap_n,
    ROUND(COALESCE(SUM(e.price_usd) FILTER (WHERE e.event = 'iap_purchase'), 0), 2) AS iap_rev
FROM s JOIN events e USING (session_id)
GROUP BY s.player_id, s.day
ORDER BY s.player_id, s.day
"""

COMPARED = ["sessions", "attempts", "completes", "continues", "interstitials", "rewarded", "ad_rev", "iap_n", "iap_rev"]


def player_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("daily", daily)
    return con.execute(PLAYER_METRICS).df()


def events_to_days(events: pd.DataFrame, t0: int) -> pd.DataFrame:
    con = duckdb.connect()
    ev = events.copy()
    for col in ("result", "format", "price_usd", "revenue_usd"):
        if col not in ev:
            ev[col] = None
    con.register("events", ev)
    return con.execute(EVENTS_TO_DAYS.format(t0=t0)).df()


def reconcile(from_events: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Player-days where the events do not add up to the reference table, metric by metric."""
    ref = daily[daily["active"]][["player_id", "day"] + COMPARED]
    m = ref.merge(from_events, on=["player_id", "day"], how="outer", suffixes=("_ref", "_ev"), indicator=True)
    out = {"player_days": int(len(ref)), "missing_in_events": int((m["_merge"] == "left_only").sum()),
           "only_in_events": int((m["_merge"] == "right_only").sum())}
    both = m[m["_merge"] == "both"]
    bad = pd.Series(False, index=both.index)
    for c in COMPARED:
        diff = (both[f"{c}_ref"].astype(float) - both[f"{c}_ev"].astype(float)).abs() > 1e-6
        out[f"mismatch_{c}"] = int(diff.sum())
        bad |= diff
    out["mismatched_player_days"] = int(bad.sum()) + out["missing_in_events"] + out["only_in_events"]
    return out
