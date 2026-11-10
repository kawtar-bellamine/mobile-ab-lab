"""Experiment readout and the checks that make it trustworthy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S

PRIMARY = ("ad_rev", "pre_ad_rev")
GUARDRAIL = ("returned_week2", "pre_returned_week", 0.01)          # must not drop by more than 1 point
SECONDARY = {"revenue": "pre_revenue", "iap_rev": "pre_iap_rev", "active_days": "pre_active_days",
             "sessions": "pre_sessions", "levels_completed": "pre_levels_completed"}


def readout(m: pd.DataFrame, treated: np.ndarray, truth: dict | None = None, boot: bool = True) -> dict:
    c, t = ~treated, treated
    out = {"n_control": int(c.sum()), "n_treatment": int(t.sum()),
           "srm_p": S.srm(int(c.sum()), int(t.sum()))}
    y, x = m[PRIMARY[0]].to_numpy(float), m[PRIMARY[1]].to_numpy(float)
    yc = S.cuped(y, x)
    out["primary"] = {"metric": PRIMARY[0], "raw": S.diff_test(y[c], y[t]), "cuped": S.diff_test(yc[c], yc[t]),
                      "variance_reduction": float(1 - yc.var() / y.var())}
    if boot:
        out["primary"]["bootstrap_ci_raw"] = S.bootstrap_ci(y[c], y[t], n_boot=1000)
    g, gx, margin = GUARDRAIL
    yg = S.cuped(m[g].to_numpy(float), m[gx].to_numpy(float))
    out["guardrail"] = {"metric": g, **S.non_inferiority(yg[c], yg[t], margin)}
    sec = {}
    for k, cov in SECONDARY.items():
        ys = S.cuped(m[k].to_numpy(float), m[cov].to_numpy(float))
        sec[k] = S.diff_test(ys[c], ys[t])
    keep = S.holm({k: v["p"] for k, v in sec.items()})
    for k in sec:
        sec[k]["significant_holm"] = keep[k]
    out["secondary"] = sec
    if truth:
        out["truth"] = truth
    p, gr = out["primary"]["cuped"], out["guardrail"]
    if out["srm_p"] < 0.001:
        out["decision"] = "INVALID: sample ratio mismatch, fix the assignment or logging before reading results"
    elif p["p"] < 0.05 and p["diff"] > 0 and gr["passes"]:
        out["decision"] = "SHIP: primary metric up, guardrail within its margin"
    elif p["p"] < 0.05 and p["diff"] > 0:
        out["decision"] = "DO NOT SHIP AS IS: primary metric up, but the retention guardrail is breached"
    else:
        out["decision"] = "NO DECISION: primary metric not significantly up"
    return out


def daily_effect(daily_obs: pd.DataFrame, treated_by_player: pd.Series) -> pd.DataFrame:
    """Ad revenue per player and per day, difference treatment minus control, from day 0 (novelty check)."""
    d = daily_obs[daily_obs["day"] >= 0].copy()
    d["treated"] = d["player_id"].map(treated_by_player)
    rows = []
    for day, g in d.groupby("day"):
        r = S.diff_test(g.loc[~g["treated"], "ad_rev"].to_numpy(), g.loc[g["treated"], "ad_rev"].to_numpy())
        rows.append({"day": int(day), "lift": r["lift"], "lo": r["lift_ci"][0], "hi": r["lift_ci"][1],
                     "control": r["control"]})
    return pd.DataFrame(rows)


def segments(m: pd.DataFrame, treated: np.ndarray) -> list[dict]:
    """Exploratory: effect for players who already watched rewarded ads before the test, and for the others."""
    rows = []
    seg = m["pre_rewarded"].to_numpy() > 0
    for name, mask in (("watched rewarded ads before the test", seg), ("never watched one", ~seg)):
        mm = m[mask]
        tt = treated[mask]
        r_ad = S.diff_test(mm.loc[~tt, "ad_rev"].to_numpy(float), mm.loc[tt, "ad_rev"].to_numpy(float))
        r_ret = S.diff_test(mm.loc[~tt, "returned_week2"].to_numpy(float), mm.loc[tt, "returned_week2"].to_numpy(float))
        rows.append({"segment": name, "players": int(mask.sum()), "ad_rev_lift": r_ad["lift"],
                     "ad_rev_lift_ci": r_ad["lift_ci"], "retention_diff": r_ret["diff"], "retention_ci": r_ret["ci"]})
    return rows


def aa_and_coverage(m0: pd.DataFrame, m1: pd.DataFrame, n_rep: int, seed: int) -> dict:
    """Random splits of the same population.
    A/A: outcomes without the feature for everyone, so every significant result is a false positive.
    Coverage: treated half gets its outcome with the feature; the known true effect must sit in the 95 % CI."""
    rng = np.random.default_rng(seed)
    n = len(m0)
    y0, y1, x = (m0["ad_rev"].to_numpy(float), m1["ad_rev"].to_numpy(float), m0["pre_ad_rev"].to_numpy(float))
    r0, r1, rx = (m0["returned_week2"].to_numpy(float), m1["returned_week2"].to_numpy(float),
                  m0["pre_returned_week"].to_numpy(float))
    v0 = m0["revenue"].to_numpy(float)
    true_ad, true_ret = float((y1 - y0).mean()), float((r1 - r0).mean())
    keys = ["aa_ad_raw", "aa_ad_cuped", "aa_ret", "aa_revenue", "cov_ad_raw", "cov_ad_cuped", "cov_ret",
            "pow_ad_raw", "pow_ad_cuped", "pow_ret"]
    res = {k: [] for k in keys}
    p_aa = []
    for _ in range(n_rep):
        t = np.zeros(n, bool)
        t[rng.permutation(n)[: n // 2]] = True
        a = S.diff_test(y0[~t], y0[t]); p_aa.append(a["p"])
        res["aa_ad_raw"].append(a["p"] < 0.05)
        yc = S.cuped(y0, x)
        res["aa_ad_cuped"].append(S.diff_test(yc[~t], yc[t])["p"] < 0.05)
        res["aa_ret"].append(S.diff_test(r0[~t], r0[t])["p"] < 0.05)
        res["aa_revenue"].append(S.diff_test(v0[~t], v0[t])["p"] < 0.05)
        yo = np.where(t, y1, y0)
        ro = np.where(t, r1, r0)
        b = S.diff_test(yo[~t], yo[t])
        yoc = S.cuped(yo, x)
        bc = S.diff_test(yoc[~t], yoc[t])
        rr = S.diff_test(S.cuped(ro, rx)[~t], S.cuped(ro, rx)[t])
        res["cov_ad_raw"].append(b["ci"][0] <= true_ad <= b["ci"][1])
        res["cov_ad_cuped"].append(bc["ci"][0] <= true_ad <= bc["ci"][1])
        res["cov_ret"].append(rr["ci"][0] <= true_ret <= rr["ci"][1])
        res["pow_ad_raw"].append(b["p"] < 0.05)
        res["pow_ad_cuped"].append(bc["p"] < 0.05)
        res["pow_ret"].append(rr["p"] < 0.05)
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["n_rep"] = n_rep
    out["aa_p_values"] = [float(p) for p in p_aa]
    return out


def peeking(daily0: pd.DataFrame, n_sim: int, seed: int, looks: int = 14) -> dict:
    """A/A experiments read every day for 14 days. Stopping at the first p < 0.05 inflates false positives;
    O'Brien-Fleming boundaries keep the overall rate at 5 %."""
    d = daily0[daily0["day"] >= 0].pivot(index="player_id", columns="day", values="ad_rev").to_numpy(float)
    cum = np.cumsum(d, axis=1)[:, :looks]
    cum2 = cum ** 2
    n = cum.shape[0]
    rng = np.random.default_rng(seed)
    bounds = S.obrien_fleming(looks)
    fixed = S.Z.ppf(0.975)
    naive_any, obf_any, final_only = 0, 0, 0
    first_naive = np.zeros(looks)
    tot, tot2 = cum.sum(0), cum2.sum(0)
    for start in range(0, n_sim, 200):
        k = min(200, n_sim - start)
        L = np.zeros((k, n))
        for i in range(k):
            L[i, rng.permutation(n)[: n // 2]] = 1
        nt = L.sum(1)[:, None]
        nc = n - nt
        st, st2 = L @ cum, L @ cum2
        sc, sc2 = tot - st, tot2 - st2
        mt, mc = st / nt, sc / nc
        vt = (st2 - nt * mt ** 2) / (nt - 1)
        vc = (sc2 - nc * mc ** 2) / (nc - 1)
        z = np.abs((mt - mc) / np.sqrt(vt / nt + vc / nc))
        hit = z > fixed
        naive_any += hit.any(1).sum()
        final_only += hit[:, -1].sum()
        obf_any += (z > bounds).any(1).sum()
        first = np.where(hit.any(1), hit.argmax(1), -1)
        for f in first[first >= 0]:
            first_naive[f] += 1
    cum_rate = np.cumsum(first_naive) / n_sim
    return {"looks": looks, "n_sim": n_sim, "fixed_horizon_fpr": final_only / n_sim, "peeking_fpr": naive_any / n_sim,
            "obrien_fleming_fpr": obf_any / n_sim, "obf_boundaries": bounds.tolist(),
            "peeking_fpr_by_look": cum_rate.tolist()}
