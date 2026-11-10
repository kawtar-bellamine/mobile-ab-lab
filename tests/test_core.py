"""Fast tests on small populations: statistics, assignment, tracking and SQL."""
from __future__ import annotations

import numpy as np

from abtest import events as E
from abtest import metrics as M
from abtest import stats as S
from abtest.game import population, simulate


def test_assignment_is_deterministic_and_balanced():
    ids = np.arange(1, 20_001)
    a, b = S.assign(ids, "exp"), S.assign(ids, "exp")
    assert (a == b).all()
    assert abs(a.mean() - 0.5) < 0.02
    assert (S.assign(ids, "other") != a).mean() > 0.4          # another experiment reshuffles players


def test_srm_flags_a_broken_split_only():
    assert S.srm(50_000, 50_100) > 0.05
    assert S.srm(50_000, 48_500) < 0.001


def test_diff_test_and_ci_on_known_data():
    rng = np.random.default_rng(0)
    c, t = rng.normal(10, 2, 50_000), rng.normal(10.2, 2, 50_000)
    r = S.diff_test(c, t)
    assert r["ci"][0] < 0.2 < r["ci"][1] and r["p"] < 1e-6
    assert abs(r["lift"] - 0.02) < 0.005


def test_cuped_keeps_the_mean_and_reduces_variance():
    rng = np.random.default_rng(1)
    x = rng.gamma(2, 1, 20_000)
    y = 0.8 * x + rng.normal(0, 0.5, 20_000)
    yc = S.cuped(y, x)
    assert abs(yc.mean() - y.mean()) < 1e-9
    assert yc.var() < 0.3 * y.var()


def test_holm_is_stricter_than_uncorrected():
    keep = S.holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert keep == {"a": True, "b": True, "c": True}
    assert S.holm({"a": 0.03, "b": 0.04})["a"] is False


def test_sample_size_matches_the_textbook_formula():
    n = S.sample_size(sd=1.0, mde_abs=0.1)
    assert abs(n - 1570) <= 2                                    # 2 * (1.96 + 0.84)^2 / 0.01
    assert S.sample_size(1.0, 0.1, rho=0.8) < 0.37 * n


def test_obrien_fleming_boundaries_are_decreasing_and_end_near_2():
    b = S.obrien_fleming(5, n_sim=50_000)
    assert all(np.diff(b) < 0) and 1.9 < b[-1] < 2.2


def test_potential_outcomes_share_the_pre_period():
    P = population(400, 1)
    y0 = simulate(P, np.zeros(400, bool), 2)
    y1 = simulate(P, np.ones(400, bool), 2)
    pre = y0["day"] < 0
    assert y0[pre].equals(y1[pre])                              # nobody sees the feature before day 0
    assert y1.loc[~pre, "continues"].sum() > 0 and y0["continues"].sum() == 0


def test_events_rebuild_the_daily_table_and_bugs_are_caught():
    P = population(120, 3)
    tr = np.arange(120) % 2 == 0
    d = simulate(P, tr, 4)
    ev = E.generate(d, P, dict(zip(P["player_id"], tr)), 5)
    assert E.validate(ev) == {k: set() for k in ("schema", "duplicate", "orphan", "exposure leak")}
    assert M.reconcile(M.events_to_days(E.to_frame(ev), E.T0), d)["mismatched_player_days"] == 0
    bad, truth = E.inject(ev, 5, 6)
    flagged = set().union(*E.validate(bad).values())
    for ids in truth.values():
        assert ids <= flagged


def test_player_metrics_sql_matches_pandas():
    P = population(200, 7)
    d = simulate(P, np.zeros(200, bool), 8)
    m = M.player_metrics(d)
    w = d[d["day"] >= 0].groupby("player_id")
    assert np.allclose(m["ad_rev"].to_numpy(), w["ad_rev"].sum().to_numpy())
    assert np.allclose(m["active_days"].to_numpy(), w["active"].sum().to_numpy())
