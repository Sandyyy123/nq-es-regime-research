"""Offline unit checks on synthetic prices (no network needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import regimes as rg
import strategies as st


def _prices(n: int = 400, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(4000.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))), index=idx)


def test_signal_reaches_returns_with_one_day_execution_lag():
    idx = pd.bdate_range("2020-01-01", periods=10)
    sig = pd.Series(0.0, index=idx)
    sig.iloc[3] = 1.0  # signal known at close of day 3
    pos = bt.lag_positions(sig)
    # Filled at close of day 4, so the first return earned is day 5 (close 4 -> close 5).
    assert bt.EXEC_LAG == 1
    assert pos.iloc[:5].eq(0.0).all()
    assert pos.iloc[5] == 1.0
    assert pos.iloc[6:].eq(0.0).all()


def test_costs_reduce_returns_by_turnover_times_cost():
    close = _prices()
    ret = close.pct_change().fillna(0.0)
    feat = rg.compute_features(close)
    pos = bt.lag_positions(st.meanrev_signal(feat))
    cost = bt.cost_rate_per_side(close, "ES")
    gross = bt.net_returns(pos, ret, cost * 0.0)
    net = bt.net_returns(pos, ret, cost)
    turnover = pos.diff().abs().fillna(pos.abs())
    assert (cost > 0).all()
    assert (net <= gross + 1e-15).all()
    assert turnover.sum() > 0 and net.sum() < gross.sum()
    np.testing.assert_allclose((gross - net).to_numpy(), (turnover * cost).to_numpy(), atol=1e-15)


def test_features_signals_and_filtered_probs_ignore_future_data():
    close = _prices(500, seed=1)
    T = close.index[350]
    full = rg.compute_features(close)
    trunc = rg.compute_features(close[close.index <= T])
    np.testing.assert_allclose(trunc.loc[T].to_numpy(float), full.loc[T].to_numpy(float), equal_nan=True)
    for name, fn in st.SLEEVES.items():
        assert fn(trunc).loc[T] == fn(full).loc[T], name

    ok = rg.valid_mask(full)
    X = full.loc[ok, rg.HMM_FEATURES]
    X = ((X - X.mean()) / X.std()).to_numpy()
    params = rg.fit_hmm_bic(X[:150], k_range=(2,), seeds=(0,))[0]
    t = 100
    P_full = rg.forward_filter(params, X)
    X_future_changed = X.copy()
    X_future_changed[t + 1:] = X_future_changed[t + 1:] * 5.0 + 3.0
    P_changed = rg.forward_filter(params, X_future_changed)
    np.testing.assert_allclose(P_full[: t + 1], P_changed[: t + 1], atol=1e-12)


def test_max_drawdown_counts_loss_from_starting_capital():
    # Equity 1.0 -> 0.90 -> 0.945 -> 0.9261: the worst fall is the first-day loss from 1.0.
    r = pd.Series([-0.10, 0.05, -0.02])
    assert abs(bt.max_drawdown(r) - (-0.10)) < 1e-12
