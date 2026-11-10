"""A/B test dashboard: plan a test, read the result, see why peeking is dangerous.

    streamlit run app.py      (after `python run.py`)
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from abtest import stats as S

ROOT = Path(__file__).parent
st.set_page_config(page_title="Mobile A/B lab", layout="wide")
R = json.loads((ROOT / "results" / "results.json").read_text())
rd, pl = R["readout"], R["plan"]

st.title("Mobile A/B lab")
st.caption("Tower Stack (fictional hyper-casual game): does a rewarded continue after a fail pay off? "
           "Simulated data with known true effects.")
p, g = rd["primary"]["cuped"], rd["guardrail"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ad revenue per player (primary)", f"{100 * p['lift']:+.1f} %")
c1.caption(f"95 % interval {100 * p['lift_ci'][0]:+.1f} % to {100 * p['lift_ci'][1]:+.1f} %")
c2.metric("Week-2 retention (guardrail)", f"{100 * g['diff']:+.1f} pt")
c2.caption(("**:red[Breached]**" if not g["passes"] else "Within margin") + ": allowed loss 1 pt")
c3.metric("Sample ratio check", f"p = {rd['srm_p']:.2f}")
c3.caption("Split as planned" if rd["srm_p"] > 0.001 else "**:red[Mismatch]**")
SHORT = {"SHIP": "Ship", "DO NOT SHIP AS IS": "Do not ship", "INVALID": "Invalid test", "NO DECISION": "No decision"}
head, _, why = rd["decision"].partition(": ")
c4.metric("Decision", SHORT.get(head, head))
c4.caption(why.capitalize())

t1, t2, t3, t4 = st.tabs(["Readout", "Plan a test", "Peeking", "Tracking"])
with t1:
    names = {"revenue": "total revenue", "iap_rev": "in-app purchases", "active_days": "active days",
             "sessions": "sessions", "levels_completed": "levels completed"}
    rows = [{"metric": "ad revenue (primary)", **p}] + [{"metric": names[k], **v} for k, v in rd["secondary"].items()]
    df = pd.DataFrame([{"metric": r_["metric"], "lift": r_["lift"], "lo": r_["lift_ci"][0], "hi": r_["lift_ci"][1]}
                       for r_ in rows])
    base = alt.Chart(df).encode(y=alt.Y("metric:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)))
    st.altair_chart((base.mark_rule(strokeWidth=3).encode(x=alt.X("lo:Q", axis=alt.Axis(format="%"), title="lift"),
                                                           x2="hi:Q")
                     + base.mark_circle(size=120).encode(x="lift:Q", tooltip=["metric", alt.Tooltip("lift", format=".1%"),
                                                                             alt.Tooltip("lo", format=".1%"),
                                                                             alt.Tooltip("hi", format=".1%")])
                     ).properties(height=260), use_container_width=True)
    d = pd.DataFrame(rd["daily"])
    band = alt.Chart(d).mark_area(opacity=0.2).encode(x=alt.X("day:Q", title="day of the experiment", axis=alt.Axis(tickMinStep=1, format="d")),
                                                      y=alt.Y("lo:Q", axis=alt.Axis(format="%"), title="ad revenue lift"),
                                                      y2="hi:Q")
    st.altair_chart((band + alt.Chart(d).mark_line(point=True).encode(x="day:Q", y="lift:Q")).properties(
        height=240, title="Novelty: the gain fades"), use_container_width=True)
    st.markdown((ROOT / "docs" / "readout.md").read_text().split("## Reading", 1)[1])
with t2:
    st.subheader("How many players does the test need?")
    mde = st.slider("Smallest lift on ad revenue worth detecting (%)", 1.0, 15.0, 3.0, 0.5)
    power = st.select_slider("Power", [0.7, 0.8, 0.9], 0.8)
    use_cuped = st.toggle("Use the pre-period (CUPED)", True)
    n = S.sample_size(pl["sd"], mde / 100 * pl["mean"], power=power, rho=pl["rho_pre_period"] if use_cuped else 0)
    st.metric("Players per arm", f"{n:,}".replace(",", " "))
    dau = st.number_input("Daily active players entering the test", 1000, 500_000, 20_000, 1000)
    st.write(f"About **{int(np.ceil(2 * n / dau))} days** of traffic, then run whole weeks (at least 14 days) "
             "so weekday and weekend players are both represented.")
with t3:
    pk = R["peeking"]
    st.subheader(f"Reading an A/A test every day: {100 * pk['peeking_fpr']:.0f} % false positives instead of 5 %")
    s = pd.DataFrame({"look": range(1, pk["looks"] + 1), "false positives": pk["peeking_fpr_by_look"],
                      "O'Brien-Fleming |z| boundary": pk["obf_boundaries"]})
    st.altair_chart(alt.Chart(s).mark_line(point=True, color="#eb6834").encode(
        x="look:Q", y=alt.Y("false positives:Q", axis=alt.Axis(format="%"))).properties(height=260),
        use_container_width=True)
    st.write(f"With O'Brien-Fleming boundaries the overall rate stays at {100 * pk['obrien_fleming_fpr']:.1f} %. "
             "Boundaries per look:")
    st.dataframe(s[["look", "O'Brien-Fleming |z| boundary"]].round(2), hide_index=True)
with t4:
    tr = R["tracking"]
    st.subheader("Tracking plan validation before any analysis")
    st.dataframe(pd.DataFrame([{"bug injected": k, "caught": v["caught"], "injected": v["injected"]}
                               for k, v in tr["detection"].items()]), hide_index=True)
    st.write(f"{tr['events']:,} events from {tr['players']:,} players. Clean events rebuild every player-day "
             f"({tr['reconciliation_clean']} mismatch); with the bugs, {tr['reconciliation_with_bugs']} player-days "
             f"are wrong; after quarantine, {tr['reconciliation_after_quarantine']} remain incomplete because the bad "
             "events had to be dropped: the fix belongs in the game client.")
    st.json(json.loads((ROOT / "data" / "tracking_plan.json").read_text()), expanded=False)
