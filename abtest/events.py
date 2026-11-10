"""Raw event log, tracking-plan validation and bug injection.

The daily table of a sample of players is expanded into the events the game would send. The validator checks
every event against the tracking plan and against the session logic; injected bugs measure what it catches.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

from .game import ECPM_INTERSTITIAL, REWARDED_FACTOR

PLAN = json.loads((Path(__file__).resolve().parents[1] / "data" / "tracking_plan.json").read_text())
T0 = 1_767_225_600            # day 0 at 00:00 UTC (fictional calendar)
EXPERIMENT = "continue_after_fail_v1"
COMMON = {"event_id": {"type": "string"}, "event": {"type": "string"}, "player_id": {"type": "integer"},
          "ts": {"type": "integer"}, "session_id": {"type": "string"}, "app_version": {"type": "string"}}


def _schema(name: str) -> dict:
    spec = PLAN["events"][name]
    return {"type": "object", "properties": {**COMMON, **spec["properties"]},
            "required": PLAN["common_required"] + spec["required"]}


VALIDATORS = {name: jsonschema.Draft7Validator(_schema(name)) for name in PLAN["events"]}


def generate(daily: pd.DataFrame, players: pd.DataFrame, treated: dict[int, bool], seed: int) -> list[dict]:
    """Expand player-day totals into ordered events, so that aggregating the events gives the totals back."""
    rng = np.random.default_rng(seed)
    tier = players.set_index("player_id")["tier"].to_dict()
    price = players.set_index("player_id")["price"].to_dict()
    events: list[dict] = []
    exposed: set[int] = set()
    k = 0

    def emit(e):
        nonlocal k
        k += 1
        e["event_id"] = f"e{k:08d}"
        events.append(e)

    for r in daily[daily["active"]].itertuples(index=False):
        pid, t = int(r.player_id), tier[int(r.player_id)]
        e_int = ECPM_INTERSTITIAL[t] / 1000
        s = int(r.sessions)
        att = 1 + rng.multinomial(r.attempts - s, np.ones(s) / s)               # attempts per session
        comp_flags = np.zeros(r.attempts, bool)
        comp_flags[rng.choice(r.attempts, r.completes, replace=False)] = True
        cont_s = np.zeros(s, int)
        if r.continues:
            cont_s[rng.choice(s, r.continues, replace=False)] = 1
        inter = rng.multinomial(r.interstitials, np.ones(s) / s)
        rew = rng.multinomial(r.rewarded - r.continues, np.ones(s) / s)
        iap = rng.multinomial(r.iap_n, np.ones(s) / s)
        ts = T0 + int(r.day) * 86_400 + int(rng.integers(8 * 3600, 12 * 3600))
        a = 0
        for j in range(s):
            sid = f"p{pid}-d{int(r.day)}-s{j + 1}"
            base = {"player_id": pid, "session_id": sid, "app_version": "3.4.1"}
            start = ts
            emit({**base, "event": "session_start", "ts": ts, "country_tier": t})
            if r.day >= 0 and pid not in exposed:
                exposed.add(pid)
                emit({**base, "event": "exp_exposure", "ts": ts, "experiment": EXPERIMENT,
                      "variant": "continue" if treated[pid] else "control"})
            for n in range(att[j]):
                a += 1
                ts += int(rng.integers(15, 60))
                if cont_s[j] and n == 0:                                           # continue on the first attempt
                    emit({**base, "event": "continue_used", "ts": ts, "attempt_no": a})
                    emit({**base, "event": "ad_impression", "ts": ts, "format": "rewarded",
                          "revenue_usd": round(e_int * REWARDED_FACTOR, 6)})
                emit({**base, "event": "level_end", "ts": ts, "attempt_no": a,
                      "result": "complete" if comp_flags[a - 1] else "fail"})
            for _ in range(inter[j]):
                ts += 1
                emit({**base, "event": "ad_impression", "ts": ts, "format": "interstitial",
                      "revenue_usd": round(e_int, 6)})
            for _ in range(rew[j]):
                ts += 1
                emit({**base, "event": "ad_impression", "ts": ts, "format": "rewarded",
                      "revenue_usd": round(e_int * REWARDED_FACTOR, 6)})
            for _ in range(iap[j]):
                ts += 1
                emit({**base, "event": "iap_purchase", "ts": ts, "price_usd": float(price[pid])})
            ts += 2
            emit({**base, "event": "session_end", "ts": ts, "duration_s": ts - start})
            ts += int(rng.integers(300, 1800))                                      # gap between sessions
    return events


BUGS = ["missing field", "wrong type", "duplicate", "orphan event", "revenue in micros", "exposure leak"]


def inject(events: list[dict], n_each: int, seed: int) -> tuple[list[dict], dict[str, set[str]]]:
    """Copy of the log with n_each bugs of every kind. Returns the log and the ids that carry each bug."""
    rng = np.random.default_rng(seed)
    ev = [dict(e) for e in events]
    truth: dict[str, set[str]] = {b: set() for b in BUGS}
    variant = {e["player_id"]: e["variant"] for e in ev if e["event"] == "exp_exposure"}
    used: set[int] = set()

    def pick(cond) -> list[int]:
        idx = [i for i, e in enumerate(ev) if cond(e) and i not in used]
        chosen = rng.choice(len(idx), n_each, replace=False)
        out = [idx[c] for c in chosen]
        used.update(out)
        return out

    for i in pick(lambda e: e["event"] == "level_end"):
        del ev[i]["result"]; truth["missing field"].add(ev[i]["event_id"])
    for i in pick(lambda e: e["event"] == "ad_impression"):
        ev[i]["revenue_usd"] = str(ev[i]["revenue_usd"]); truth["wrong type"].add(ev[i]["event_id"])
    for i in pick(lambda e: e["event"] in ("level_end", "ad_impression")):
        ev[i]["session_id"] = ev[i]["session_id"] + "x"; truth["orphan event"].add(ev[i]["event_id"])
    for i in pick(lambda e: e["event"] == "ad_impression" and isinstance(e["revenue_usd"], float)):
        ev[i]["revenue_usd"] = round(ev[i]["revenue_usd"] * 1e6, 1); truth["revenue in micros"].add(ev[i]["event_id"])
    dups = pick(lambda e: e["event"] in ("level_end", "ad_impression", "iap_purchase"))
    for i in dups:
        truth["duplicate"].add(ev[i]["event_id"])
    ev += [dict(ev[i]) for i in dups]                                           # resent by a retry
    for j, i in enumerate(pick(lambda e: e["event"] == "level_end" and variant.get(e["player_id"]) == "control"
                               and not e["session_id"].endswith("x") and "-d-" not in e["session_id"]
                               and int(e["session_id"].split("-d")[1].split("-")[0]) >= 0)):
        leak = {**ev[i], "event": "continue_used", "event_id": f"leak{j:05d}"}
        leak.pop("result", None)
        ev.append(leak); truth["exposure leak"].add(leak["event_id"])
    ev.sort(key=lambda e: (e["player_id"], e["ts"]))
    return ev, truth


def validate(events: list[dict]) -> dict[str, set[str]]:
    """Every rule returns the ids it flags. Rules do not know which bugs were injected."""
    flags: dict[str, set[str]] = {r: set() for r in ("schema", "duplicate", "orphan", "exposure leak")}
    seen: Counter = Counter()
    sessions = {e["session_id"] for e in events if e.get("event") == "session_start"}
    variant = {e["player_id"]: e.get("variant") for e in events if e.get("event") == "exp_exposure"}
    for e in events:
        eid = e.get("event_id", "?")
        v = VALIDATORS.get(e.get("event"))
        if v is None or next(v.iter_errors(e), None) is not None:
            flags["schema"].add(eid)
        seen[eid] += 1
        if seen[eid] > 1:
            flags["duplicate"].add(eid)
        if e.get("session_id") not in sessions:
            flags["orphan"].add(eid)
        if e.get("event") == "continue_used" and variant.get(e.get("player_id")) != "continue":
            flags["exposure leak"].add(eid)
    return flags


def to_frame(events: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(events)
    df["revenue_usd"] = pd.to_numeric(df.get("revenue_usd"), errors="coerce")
    return df


def quarantine(events: list[dict], flags: dict[str, set[str]]) -> list[dict]:
    """Keep the first copy of a resent event, drop every other flagged event."""
    drop = flags["schema"] | flags["orphan"] | flags["exposure leak"]
    seen, out = set(), []
    for e in events:
        eid = e.get("event_id")
        if eid in drop or eid in seen:
            continue
        seen.add(eid)
        out.append(e)
    return out
