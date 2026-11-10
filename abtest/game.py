"""Simulated hyper-casual mobile game, with known potential outcomes.

"Tower Stack" is a fictional one-tap game: a player makes attempts, each attempt ends in a completed level or
a fail. The experiment tests a rewarded "continue" offered after a fail (watch an ad, keep the tower).

Every player is simulated under BOTH arms with common random numbers, so the true effect of the feature on
this exact population is known. That is what lets the repository check its own statistics: does the 95 %
interval contain the truth 95 % of the time, does an A/A test fire 5 % of the time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRE_DAYS, TEST_DAYS = 14, 14
DAYS = np.arange(-PRE_DAYS, TEST_DAYS)                 # day 0 = start of the experiment
TIERS = np.array(["T1", "T2", "T3"])                   # country tiers, by ad value
TIER_P = [0.25, 0.35, 0.40]
ECPM_INTERSTITIAL = {"T1": 14.0, "T2": 6.0, "T3": 1.8}  # $ per 1,000 impressions
REWARDED_FACTOR = 1.8                                  # rewarded ads pay more than interstitials
PRICE_POINTS = np.array([0.99, 2.99, 4.99, 9.99, 19.99, 49.99])
PRICE_P = np.array([0.30, 0.25, 0.20, 0.13, 0.08, 0.04])


@dataclass(frozen=True)
class Feature:
    """Rewarded continue after a fail. Effects are mechanisms, not a target number."""
    novelty_boost: float = 0.15       # extra sessions at launch, fading with a 2-day half-life
    fatigue: float = 0.15             # lower daily activity for players who dislike ads
    completes_per_continue: float = 0.6


def population(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tier = rng.choice(TIERS, n, p=TIER_P)
    payer_p = np.select([tier == "T1", tier == "T2"], [0.04, 0.025], 0.01)
    return pd.DataFrame({
        "player_id": np.arange(1, n + 1),
        "tier": tier,
        "engagement": rng.gamma(1.5, 1.0, n),
        "weekly_survival": rng.beta(9, 1.5, n),
        "skill": rng.beta(6, 4, n),
        "ad_tolerance": rng.beta(3, 2, n),
        "payer": rng.random(n) < payer_p,
        "price": rng.choice(PRICE_POINTS, n, p=PRICE_P),
    })


def simulate(players: pd.DataFrame, treated: np.ndarray, seed: int, feature: Feature = Feature()) -> pd.DataFrame:
    """One row per player and day. `treated` is a boolean array; the feature only acts from day 0.

    The same seed gives the same random draws whatever the arm, so simulate(..., all False) and
    simulate(..., all True) are the two potential outcomes of every player.
    """
    rng = np.random.default_rng(seed)
    n, d = len(players), len(DAYS)
    eng = players["engagement"].to_numpy()[:, None]
    surv = players["weekly_survival"].to_numpy()[:, None]
    skill = players["skill"].to_numpy()[:, None]
    tol = players["ad_tolerance"].to_numpy()[:, None]
    on = treated[:, None] & (DAYS[None, :] >= 0)                          # feature visible
    age = (DAYS + PRE_DAYS)[None, :] / 7.0

    # draws shared by both arms
    u_active, u_sess, u_att, u_comp, u_acc, u_rew, u_iap = (rng.random((n, d)) for _ in range(7))
    z_att = rng.random((n, d))

    p_active = (1 - np.exp(-0.35 * eng)) * surv ** age
    p_active = p_active * np.where(on, 1 - feature.fatigue * (1 - tol), 1.0)
    active = u_active < p_active

    from scipy.stats import binom, poisson
    novelty = 1 + feature.novelty_boost * np.exp(-np.clip(DAYS, 0, None)[None, :] * np.log(2) / 2)
    lam_s = 1.2 * eng * np.where(on, novelty, 1.0)
    sessions = np.where(active, 1 + poisson.ppf(np.minimum(u_sess, 1 - 1e-12), lam_s), 0).astype(int)
    attempts = np.where(sessions > 0, poisson.ppf(np.minimum(u_att, 1 - 1e-12), 6.0 * sessions) + sessions, 0)
    attempts = attempts.astype(int)
    completes = binom.ppf(u_comp, attempts, skill).astype(int)
    fails = attempts - completes
    accept_p = 0.25 * tol                                                 # one offer per session, at the first fail
    offers = np.minimum(fails, sessions)
    continues = np.where(on, binom.ppf(u_acc, offers, accept_p), 0).astype(int)
    extra = np.floor(feature.completes_per_continue * continues + z_att * 0.999).astype(int)
    completes = np.minimum(completes + extra, attempts)
    interstitials = (attempts + continues) // 3
    rewarded_base = poisson.ppf(np.minimum(u_rew, 1 - 1e-12), 0.2 * sessions).astype(int)
    rewarded = rewarded_base + continues
    ecpm = players["tier"].map(ECPM_INTERSTITIAL).to_numpy()[:, None] / 1000
    ad_rev = interstitials * ecpm + rewarded * ecpm * REWARDED_FACTOR
    iap_n = np.where(players["payer"].to_numpy()[:, None] & active,
                     poisson.ppf(np.minimum(u_iap, 1 - 1e-12), 0.05 * sessions), 0).astype(int)
    iap_rev = iap_n * players["price"].to_numpy()[:, None]

    out = pd.DataFrame({
        "player_id": np.repeat(players["player_id"].to_numpy(), d),
        "day": np.tile(DAYS, n),
        "active": active.ravel(),
        "sessions": sessions.ravel(),
        "attempts": attempts.ravel(),
        "completes": completes.ravel(),
        "fails": (attempts - completes).ravel(),
        "continues": continues.ravel(),
        "interstitials": interstitials.ravel(),
        "rewarded": rewarded.ravel(),
        "ad_rev": ad_rev.ravel().round(6),
        "iap_n": iap_n.ravel(),
        "iap_rev": iap_rev.ravel().round(2),
    })
    return out
