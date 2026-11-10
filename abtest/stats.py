"""Statistics for online experiments, written out so every step can be checked."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import stats

Z = stats.norm


def assign(player_ids, salt: str, share: float = 0.5) -> np.ndarray:
    """Deterministic assignment: the same player always lands in the same arm for a given experiment."""
    h = [int(hashlib.sha256(f"{salt}:{p}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for p in player_ids]
    return np.array(h) < share


def srm(n_control: int, n_treatment: int, share: float = 0.5) -> float:
    """Sample ratio mismatch: chi-square p-value of the observed split against the planned one."""
    n = n_control + n_treatment
    return float(stats.chisquare([n_control, n_treatment], [n * (1 - share), n * share]).pvalue)


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CUPED (Deng et al., 2013): remove the part of y explained by a pre-experiment covariate x.
    theta is estimated on both arms pooled, which keeps the estimator unbiased."""
    c = np.cov(y, x)
    theta = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 0.0
    return y - theta * (x - x.mean())


def diff_test(y_c: np.ndarray, y_t: np.ndarray, alpha: float = 0.05) -> dict:
    """Difference of means with Welch's standard error (also valid for 0/1 metrics at these sample sizes)."""
    m_c, m_t = y_c.mean(), y_t.mean()
    se = np.sqrt(y_c.var(ddof=1) / len(y_c) + y_t.var(ddof=1) / len(y_t))
    d = m_t - m_c
    z = d / se if se > 0 else 0.0
    h = Z.ppf(1 - alpha / 2) * se
    # relative lift with the delta method
    rel = d / m_c if m_c else np.nan
    se_rel = np.sqrt(y_t.var(ddof=1) / len(y_t) / m_c ** 2 + m_t ** 2 * y_c.var(ddof=1) / len(y_c) / m_c ** 4)
    return {"control": float(m_c), "treatment": float(m_t), "diff": float(d), "ci": (float(d - h), float(d + h)),
            "se": float(se), "p": float(2 * Z.sf(abs(z))), "lift": float(rel),
            "lift_ci": (float(rel - Z.ppf(1 - alpha / 2) * se_rel), float(rel + Z.ppf(1 - alpha / 2) * se_rel))}


def bootstrap_ci(y_c: np.ndarray, y_t: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap of the difference of means, for heavy-tailed revenue. Poisson weights keep it fast."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        w_c, w_t = rng.poisson(1.0, len(y_c)), rng.poisson(1.0, len(y_t))
        out[b] = (w_t @ y_t) / w_t.sum() - (w_c @ y_c) / w_c.sum()
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def non_inferiority(y_c: np.ndarray, y_t: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """Guardrail: the metric must not drop by more than `margin` (absolute). One-sided test at level alpha."""
    r = diff_test(y_c, y_t, alpha=2 * alpha)
    lower = r["ci"][0]
    return {**r, "margin": -margin, "lower_bound": lower, "passes": bool(lower > -margin)}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni: controls the family-wise error rate over secondary metrics."""
    order = sorted(pvalues, key=pvalues.get)
    out, still = {}, True
    for i, k in enumerate(order):
        still = still and pvalues[k] <= alpha / (len(order) - i)
        out[k] = still
    return out


def sample_size(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8, rho: float = 0.0) -> int:
    """Players per arm to detect an absolute difference mde_abs. rho is the CUPED correlation (variance x 1-rho^2)."""
    var = sd ** 2 * (1 - rho ** 2)
    return int(np.ceil(2 * var * (Z.ppf(1 - alpha / 2) + Z.ppf(power)) ** 2 / mde_abs ** 2))


def obrien_fleming(k_looks: int, alpha: float = 0.05, n_sim: int = 200_000, seed: int = 0) -> np.ndarray:
    """Two-sided O'Brien-Fleming boundaries c / sqrt(t_k) for equally spaced looks, with c set by simulating
    the Brownian motion of the z statistic under the null (no dependence on the experiment's data)."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, k_looks + 1) / k_looks
    inc = rng.standard_normal((n_sim, k_looks)) * np.sqrt(np.diff(np.r_[0, t]))
    zpath = np.cumsum(inc, axis=1) / np.sqrt(t)
    scaled = np.max(np.abs(zpath) * np.sqrt(t), axis=1)       # crossing c/sqrt(t) <=> |z| sqrt(t) > c
    c = np.quantile(scaled, 1 - alpha)
    return c / np.sqrt(t)
