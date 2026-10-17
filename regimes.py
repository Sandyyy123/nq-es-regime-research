"""Feature construction and regime models (no lookahead).

Every feature at date t uses closes up to and including t only (trailing windows).
The HMM is fitted on a training window, its states are relabelled by volatility,
and regimes are read from FILTERED (forward-only) probabilities: P(state_t | obs_1..t).
Smoothed posteriors (forward-backward) and Viterbi paths are never used, because both
condition on future observations.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

TRADING_DAYS = 252
HMM_FEATURES = ["r1", "log_rv20", "r20"]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def compute_features(close: pd.Series) -> pd.DataFrame:
    """Trailing-window features. Row t depends only on close[:t+1]."""
    logp = np.log(close)
    r1 = logp.diff()
    f = pd.DataFrame(index=close.index)
    f["close"] = close
    f["r1"] = r1
    f["rv20"] = r1.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    f["log_rv20"] = np.log(f["rv20"])
    f["r5"] = logp.diff(5)
    f["r20"] = logp.diff(20)
    f["r60"] = logp.diff(60)
    # 60-day OLS slope of log price on time (annualised), closed form on rolling moments.
    t = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    w = 60
    mt = t.rolling(w).mean()
    my = logp.rolling(w).mean()
    mty = (t * logp).rolling(w).mean()
    mtt = (t * t).rolling(w).mean()
    f["slope60"] = (mty - mt * my) / (mtt - mt * mt) * TRADING_DAYS
    f["ma50"] = close.rolling(50, min_periods=50).mean()
    f["ma200"] = close.rolling(200, min_periods=200).mean()
    f["dd252"] = close / close.rolling(252, min_periods=252).max() - 1.0
    return f


def valid_mask(feat: pd.DataFrame) -> pd.Series:
    cols = HMM_FEATURES + ["rv20", "r60", "ma200", "dd252", "r5"]
    return feat[cols].notna().all(axis=1) & np.isfinite(feat["log_rv20"])


# ---------------------------------------------------------------------------
# Gaussian HMM with BIC selection, volatility-sorted labels, forward filter
# ---------------------------------------------------------------------------
def _n_params_full(k: int, d: int) -> int:
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def fit_hmm_bic(X: np.ndarray, k_range=(2, 3, 4), seeds=(0, 1, 2, 3, 4), n_iter=300):
    """Fit GaussianHMM for each K (best of several seeds by log-likelihood); pick K by BIC.

    X must be TRAINING data only. Returns (params dict, bic table).
    """
    n, d = X.shape
    table = []
    best_overall = None
    for k in k_range:
        best = None
        for s in seeds:
            m = GaussianHMM(n_components=k, covariance_type="full", n_iter=n_iter,
                            tol=1e-4, random_state=s, min_covar=1e-4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X)
                ll = m.score(X)
            if np.isfinite(ll) and (best is None or ll > best[1]):
                best = (m, ll)
        m, ll = best
        bic = -2.0 * ll + _n_params_full(k, d) * np.log(n)
        table.append({"k": k, "loglik": float(ll), "bic": float(bic),
                      "n_params": _n_params_full(k, d), "converged": bool(m.monitor_.converged)})
        if best_overall is None or bic < best_overall[1]:
            best_overall = (m, bic, k)
    m, _, k = best_overall
    params = sort_states_by_vol(m)
    params["k"] = k
    params["converged"] = bool(m.monitor_.converged)
    return params, table


def sort_states_by_vol(m: GaussianHMM) -> dict:
    """Relabel states so state 0 has the lowest daily-return variance (feature 0 = r1)."""
    covars = np.asarray(m.covars_)  # (k, d, d) full matrices
    order = np.argsort(covars[:, 0, 0])
    return {
        "startprob": np.asarray(m.startprob_)[order],
        "transmat": np.asarray(m.transmat_)[np.ix_(order, order)],
        "means": np.asarray(m.means_)[order],
        "covars": covars[order],
    }


def forward_filter(params: dict, X: np.ndarray) -> np.ndarray:
    """Filtered probabilities P(s_t | x_1..x_t). Row t uses rows 0..t of X only."""
    k = len(params["startprob"])
    logB = np.column_stack([
        multivariate_normal(mean=params["means"][j], cov=params["covars"][j],
                            allow_singular=True).logpdf(X)
        for j in range(k)
    ])
    logA = np.log(np.clip(params["transmat"], 1e-300, None))
    out = np.empty((len(X), k))
    log_alpha = np.log(np.clip(params["startprob"], 1e-300, None)) + logB[0]
    log_alpha -= logsumexp(log_alpha)
    out[0] = np.exp(log_alpha)
    for t in range(1, len(X)):
        log_pred = logsumexp(log_alpha[:, None] + logA, axis=0)
        log_alpha = log_pred + logB[t]
        log_alpha -= logsumexp(log_alpha)
        out[t] = np.exp(log_alpha)
    return out


def state_to_bucket(state_idx: np.ndarray, k: int) -> np.ndarray:
    """Map volatility-sorted states to comparable buckets across refits with different K."""
    lab = np.full(len(state_idx), "mid_vol", dtype=object)
    lab[state_idx == 0] = "low_vol"
    lab[state_idx == k - 1] = "high_vol"
    return lab


HMM_BUCKETS = ["low_vol", "mid_vol", "high_vol"]


# ---------------------------------------------------------------------------
# Rules baseline: volatility tercile (training thresholds) x trend sign
# ---------------------------------------------------------------------------
RULE_BUCKETS = [f"{v}_{t}" for v in ("lowvol", "midvol", "highvol") for t in ("up", "down")]


def rules_thresholds(train_feat: pd.DataFrame) -> tuple[float, float]:
    q = train_feat["rv20"].quantile([1 / 3, 2 / 3])
    return float(q.iloc[0]), float(q.iloc[1])


def rules_regime(feat: pd.DataFrame, q1: float, q2: float) -> pd.Series:
    vol = np.where(feat["rv20"] <= q1, "lowvol", np.where(feat["rv20"] <= q2, "midvol", "highvol"))
    trend = np.where(feat["r60"] >= 0, "up", "down")
    return pd.Series([f"{a}_{b}" for a, b in zip(vol, trend)], index=feat.index)
