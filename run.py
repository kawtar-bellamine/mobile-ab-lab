"""Run the whole study: tracking validation, the experiment readout and the checks behind it.

    python run.py          # about 1 minute on a laptop; results/, docs/figures/, docs/readout.md, exports/
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from abtest import analysis as A
from abtest import events as E
from abtest import metrics as M
from abtest import stats as S
from abtest.game import population, simulate

ROOT = Path(__file__).parent
RES, FIG, EXP = ROOT / "results", ROOT / "docs" / "figures", ROOT / "exports"
N_PLAYERS, SEED = 60_000, 2026
EXPERIMENT = "continue_after_fail_v1"


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    for d in (RES, FIG, EXP):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    players = population(N_PLAYERS, SEED)
    treated = S.assign(players["player_id"], EXPERIMENT)
    daily0 = simulate(players, np.zeros(N_PLAYERS, bool), SEED + 1)          # everyone without the feature
    daily1 = simulate(players, np.ones(N_PLAYERS, bool), SEED + 1)           # everyone with it
    tmask = np.repeat(treated, len(daily0) // N_PLAYERS)
    daily = daily0.copy()
    daily.loc[tmask] = daily1.loc[tmask]
    m0, m1, m = M.player_metrics(daily0), M.player_metrics(daily1), M.player_metrics(daily)
    truth = {k: float((m1[k] - m0[k]).mean()) for k in ["ad_rev", "returned_week2", "revenue", "iap_rev",
                                                         "active_days", "sessions", "levels_completed"]}
    truth["ad_rev_lift"] = truth["ad_rev"] / float(m0["ad_rev"].mean())
    print(f"simulated {N_PLAYERS:,} players in {time.time() - t0:.0f} s", flush=True)

    # 1. Tracking: validate the raw events of a sample of players before trusting any metric
    sample = players.iloc[:1500]
    ds = daily[daily["player_id"].isin(sample["player_id"])]
    ev = E.generate(ds, sample, dict(zip(players["player_id"], treated)), SEED + 2)
    clean_rec = M.reconcile(M.events_to_days(E.to_frame(ev), E.T0), ds)
    bad, injected = E.inject(ev, 60, SEED + 3)
    flags = E.validate(bad)
    flagged = set().union(*flags.values())
    clean_ids = {e["event_id"] for e in ev}
    raw_rec = M.reconcile(M.events_to_days(E.to_frame(bad), E.T0), ds)
    q_rec = M.reconcile(M.events_to_days(E.to_frame(E.quarantine(bad, flags)), E.T0), ds)
    tracking = {"events": len(ev), "players": len(sample), "player_days": clean_rec["player_days"],
                "reconciliation_clean": clean_rec["mismatched_player_days"],
                "reconciliation_with_bugs": raw_rec["mismatched_player_days"],
                "reconciliation_after_quarantine": q_rec["mismatched_player_days"],
                "false_alarms_on_clean_events": len((flagged - set().union(*injected.values())) & clean_ids),
                "detection": {b: {"caught": len(ids & flagged), "injected": len(ids)} for b, ids in injected.items()}}
    print("tracking", tracking["detection"], flush=True)

    # 2. The experiment readout
    read = A.readout(m, treated, truth)
    read["daily"] = A.daily_effect(daily, pd.Series(treated, index=players["player_id"])).to_dict("records")
    read["segments"] = A.segments(m, treated)
    print(read["decision"], flush=True)

    # 3. Checks on the statistics themselves
    checks = A.aa_and_coverage(m0, m1, n_rep=1000, seed=SEED + 4)
    peek = A.peeking(daily0, n_sim=2000, seed=SEED + 5)
    ctrl = m0["ad_rev"].to_numpy(float)
    rho = float(np.corrcoef(ctrl, m0["pre_ad_rev"])[0, 1])
    sd, mean = float(ctrl.std()), float(ctrl.mean())
    plan = {"sd": sd, "mean": mean, "rho_pre_period": rho,
            "mde_relative": [0.02, 0.03, 0.05, 0.08, 0.10],
            "n_per_arm_raw": [S.sample_size(sd, r * mean) for r in [0.02, 0.03, 0.05, 0.08, 0.10]],
            "n_per_arm_cuped": [S.sample_size(sd, r * mean, rho=rho) for r in [0.02, 0.03, 0.05, 0.08, 0.10]]}
    rv = m0["revenue"].to_numpy(float)
    plan["revenue_cv"] = float(rv.std() / rv.mean())
    plan["revenue_n_per_arm_5pct"] = S.sample_size(rv.std(), 0.05 * rv.mean(),
                                                   rho=float(np.corrcoef(rv, m0["pre_revenue"])[0, 1]))
    print(f"A/A {checks['aa_ad_cuped']:.3f}, coverage {checks['cov_ad_cuped']:.3f}, peeking {peek['peeking_fpr']:.3f}",
          flush=True)

    # 4. What a logging bug does: the exposure event fails on old devices, mostly for low-engagement players
    rng = np.random.default_rng(SEED + 6)
    eng = players["engagement"].to_numpy()
    lost = treated & (eng < np.quantile(eng, 0.25)) & (rng.random(N_PLAYERS) < 0.12)
    keep = ~lost
    srm_case = A.readout(m[keep].reset_index(drop=True), treated[keep], boot=False)
    bug = {"players_lost": int(lost.sum()), "srm_p": srm_case["srm_p"], "decision": srm_case["decision"],
           "retention_diff_biased": srm_case["guardrail"]["diff"], "retention_diff_true": truth["returned_week2"],
           "ad_rev_lift_biased": srm_case["primary"]["cuped"]["lift"]}

    results = jsonable({"tracking": tracking, "readout": read, "checks": checks, "peeking": peek, "plan": plan,
                        "srm_bug": bug, "players": N_PLAYERS})
    (RES / "results.json").write_text(json.dumps(results, indent=1))

    # exports for Power BI: one row per player, and daily totals per arm and tier
    pm = m.assign(arm=np.where(treated, "continue", "control"), tier=players["tier"].to_numpy())
    pm.round(6).to_csv(EXP / "players.csv", index=False)
    dd = daily.assign(arm=np.repeat(np.where(treated, "continue", "control"), len(daily) // N_PLAYERS),
                      tier=np.repeat(players["tier"].to_numpy(), len(daily) // N_PLAYERS))
    agg = dd.groupby(["day", "arm", "tier"]).agg(players=("player_id", "nunique"), active=("active", "sum"),
                                                 sessions=("sessions", "sum"), ad_rev=("ad_rev", "sum"),
                                                 iap_rev=("iap_rev", "sum"), continues=("continues", "sum"))
    agg.reset_index().round(4).to_csv(EXP / "daily_by_arm_tier.csv", index=False)

    from abtest.report import draw, write_readout
    draw(results, FIG)
    write_readout(results, ROOT / "docs" / "readout.md")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
