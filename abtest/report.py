"""Figures and the decision memo, drawn from results/results.json only."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
NEUTRAL, CRITICAL = "#9a9893", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})
PCT = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def draw(r: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rd, truth = r["readout"], r["readout"]["truth"]

    # 1. Readout: every metric with its 95 % interval, and the true effect known from the simulation
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [2.2, 1]})
    rows = [("ad revenue, CUPED (primary)", rd["primary"]["cuped"], truth["ad_rev_lift"]),
            ("ad revenue, raw", rd["primary"]["raw"], truth["ad_rev_lift"])]
    names = {"revenue": "total revenue", "iap_rev": "in-app purchases", "active_days": "active days",
             "sessions": "sessions", "levels_completed": "levels completed"}
    base = {k: rd["secondary"][k]["control"] for k in names}
    for k, label in names.items():
        tr = truth[k] / base[k] if base[k] else np.nan
        rows.append((label, rd["secondary"][k], tr))
    for i, (label, res, tr) in enumerate(rows):
        y = len(rows) - 1 - i
        col = BLUE if i == 0 else (AQUA if res["lift_ci"][0] > 0 else (CRITICAL if res["lift_ci"][1] < 0 else NEUTRAL))
        a1.plot(res["lift_ci"], [y, y], color=col, linewidth=2.5, solid_capstyle="round")
        a1.plot(res["lift"], y, "o", color=col, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        a1.plot(tr, y, "|", color=INK, markersize=14, markeredgewidth=2)
    a1.axvline(0, color=INK2, linewidth=1)
    a1.set_yticks(range(len(rows)), [r_[0] for r_ in rows][::-1])
    a1.xaxis.set_major_formatter(PCT)
    a1.set_title("Effect of the rewarded continue, 14 days, 95 % intervals")
    a1.set_xlabel("relative difference, continue vs control      | = true effect (known from the simulation)")
    a1.grid(axis="y", visible=False)
    g = rd["guardrail"]
    a2.axvspan(-0.05, g["margin"], color=CRITICAL, alpha=0.08)
    a2.axvline(g["margin"], color=CRITICAL, linewidth=1.2, linestyle="--")
    a2.plot(g["ci"], [0, 0], color=CRITICAL, linewidth=2.5, solid_capstyle="round")
    a2.plot(g["diff"], 0, "o", color=CRITICAL, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    a2.plot(truth["returned_week2"], 0, "|", color=INK, markersize=14, markeredgewidth=2)
    a2.axvline(0, color=INK2, linewidth=1)
    a2.set_xlim(-0.03, 0.01); a2.set_ylim(-1, 1); a2.set_yticks([])
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:+.0f} pt"))
    a2.set_title("Guardrail: week-2 retention")
    a2.text(g["margin"] - 0.001, 0.62, "allowed\nloss: 1 pt", ha="right", color=CRITICAL, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "readout.png", dpi=160); plt.close(fig)

    # 2. Novelty: the lift is largest at launch and fades
    d = rd["daily"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    days = [x["day"] for x in d]
    ax.fill_between(days, [x["lo"] for x in d], [x["hi"] for x in d], color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(days, [x["lift"] for x in d], color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.axhline(rd["primary"]["cuped"]["lift"], color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(13.2, rd["primary"]["cuped"]["lift"], "14-day average", color=INK2, fontsize=9, va="bottom", ha="right")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("day of the experiment"); ax.set_ylabel("ad revenue per player, lift")
    ax.set_title("Novelty effect: the ad revenue gain fades over two weeks")
    fig.tight_layout(); fig.savefig(out / "novelty.png", dpi=160); plt.close(fig)

    # 3. Are the statistics honest? A/A p-values, false positives, coverage
    c = r["checks"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.hist(c["aa_p_values"], bins=20, range=(0, 1), color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    a1.axhline(c["n_rep"] / 20, color=INK2, linestyle="--", linewidth=1)
    a1.set_title(f"{c['n_rep']} A/A tests: p-values are uniform"); a1.set_xlabel("p-value"); a1.set_ylabel("tests")
    a1.grid(axis="x", visible=False)
    labels = ["A/A false\npositives", "95 % CI covers\nthe true effect", "power with\n60,000 players"]
    vals = [c["aa_ad_cuped"], c["cov_ad_cuped"], c["pow_ad_cuped"]]
    targets = [0.05, 0.95, 0.80]
    a2.bar(range(3), vals, color=[BLUE, AQUA, YELLOW], width=0.55)
    for i, (v, t) in enumerate(zip(vals, targets)):
        a2.plot([i - 0.35, i + 0.35], [t, t], color=INK, linewidth=1.5)
        a2.text(i, v + 0.03, f"{100 * v:.1f} %", ha="center", fontsize=10)
    a2.set_xticks(range(3), labels); a2.set_ylim(0, 1.12); a2.yaxis.set_major_formatter(PCT)
    a2.set_title("Ad revenue test, CUPED (black line = target)"); a2.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=160); plt.close(fig)

    # 4. Peeking
    p = r["peeking"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    looks = np.arange(1, p["looks"] + 1)
    ax.plot(looks, p["peeking_fpr_by_look"], color=ORANGE, linewidth=2, marker="o", markersize=5,
            label="stop at the first p < 0.05")
    ax.axhline(0.05, color=INK2, linestyle="--", linewidth=1)
    ax.text(9, 0.053, "5 % promised", color=INK2, fontsize=9)
    ax.plot([p["looks"]], [p["obrien_fleming_fpr"]], "D", color=AQUA, markersize=8,
            label=f"O'Brien-Fleming boundaries: {100 * p['obrien_fleming_fpr']:.1f} %")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(PCT); ax.set_ylim(0, max(p["peeking_fpr_by_look"]) * 1.25)
    ax.set_xlabel("days the results were looked at"); ax.set_ylabel("A/A tests declared significant")
    ax.set_title(f"Looking every day turns 5 % into {100 * p['peeking_fpr']:.0f} % false positives")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "peeking.png", dpi=160); plt.close(fig)

    # 5. Sample size
    pl = r["plan"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = [100 * v for v in pl["mde_relative"]]
    ax.plot(x, pl["n_per_arm_raw"], color=NEUTRAL, linewidth=2, marker="o", label="plain comparison")
    ax.plot(x, pl["n_per_arm_cuped"], color=BLUE, linewidth=2, marker="o",
            label=f"CUPED (pre-period correlation {pl['rho_pre_period']:.2f})")
    ax.axhline(r["players"] / 2, color=INK2, linestyle="--", linewidth=1)
    ax.text(9.9, r["players"] / 2, "players per arm available", color=INK2, fontsize=9, va="bottom", ha="right")
    ax.set_yscale("log"); ax.set_xlabel("smallest lift to detect, ad revenue (%)"); ax.set_ylabel("players per arm")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("Players needed for 80 % power"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out / "sample_size.png", dpi=160); plt.close(fig)


def write_readout(r: dict, path: Path):
    rd, t, c = r["readout"], r["tracking"], r["checks"]
    p, g = rd["primary"]["cuped"], rd["guardrail"]
    d = rd["daily"]
    lines = [
        "# Readout: continue_after_fail_v1",
        "",
        "*Generated by `python run.py` from simulated data. Tower Stack is a fictional game.*",
        "",
        f"**Decision: {rd['decision']}.**",
        "",
        "## Validity",
        "",
        f"* Split: {rd['n_control']:,} control, {rd['n_treatment']:,} continue; sample ratio p = {rd['srm_p']:.2f} "
        "(alarm below 0.001).",
        f"* Tracking: {sum(v['caught'] for v in t['detection'].values())} injected bugs caught out of "
        f"{sum(v['injected'] for v in t['detection'].values())}, {t['false_alarms_on_clean_events']} false alarm on "
        f"{t['events']:,} clean events; clean events rebuild every player-day exactly.",
        f"* Statistics: {100 * c['aa_ad_cuped']:.1f} % false positives over {c['n_rep']} A/A tests, "
        f"{100 * c['cov_ad_cuped']:.1f} % of 95 % intervals contain the true effect.",
        "",
        "## Results (14 days)",
        "",
        "| Metric | Control | Continue | Lift | 95 % interval |",
        "|---|---:|---:|---:|---|",
        f"| **Ad revenue per player (primary, CUPED)** | ${p['control']:.3f} | ${p['treatment']:.3f} | "
        f"**{100 * p['lift']:+.1f} %** | {100 * p['lift_ci'][0]:+.1f} % to {100 * p['lift_ci'][1]:+.1f} % |",
        f"| **Week-2 retention (guardrail)** | {100 * g['control']:.1f} % | {100 * g['treatment']:.1f} % | "
        f"**{100 * g['diff']:+.1f} pt** | lower bound {100 * g['lower_bound']:+.1f} pt, allowed -1.0 pt |",
    ]
    names = {"revenue": "Total revenue", "iap_rev": "In-app purchases", "active_days": "Active days",
             "sessions": "Sessions", "levels_completed": "Levels completed"}
    for k, n in names.items():
        s = rd["secondary"][k]
        star = " (significant after Holm)" if s["significant_holm"] else ""
        lines.append(f"| {n}{star} | {s['control']:.3f} | {s['treatment']:.3f} | {100 * s['lift']:+.1f} % | "
                     f"{100 * s['lift_ci'][0]:+.1f} % to {100 * s['lift_ci'][1]:+.1f} % |")
    lines += [
        "",
        "## Reading",
        "",
        f"* The continue sells ads: {100 * p['lift']:+.1f} % ad revenue per player. But the gain fades, from "
        f"{100 * d[0]['lift']:+.0f} % on day 0 to {100 * np.mean([x['lift'] for x in d[-4:]]):+.0f} % over the last four "
        "days, while players come back less: active days and sessions drop, and week-2 retention loses "
        f"{-100 * g['diff']:.1f} points, beyond the 1-point margin agreed before the test.",
        "* Total revenue is not significant: in-app purchases make it too noisy to read in two weeks "
        f"(coefficient of variation {r['plan']['revenue_cv']:.1f}, {r['plan']['revenue_n_per_arm_5pct']:,} players per "
        "arm to detect 5 %).",
        "* Exploratory segments do not show a group where retention holds; they are hypotheses, not a reason to ship.",
        "",
        "## Next steps",
        "",
        "1. Do not roll out the continue as it is.",
        "2. Test a capped version (one continue every other session) against control, 4 weeks, same guardrail.",
        "3. Keep a 5 % long-term holdout on any version that ships, to measure the effect once the novelty is gone.",
        "",
    ]
    path.write_text("\n".join(lines))
