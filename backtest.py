"""Backtest mechanics: execution lag, transaction costs, metrics, block bootstrap.

Timing convention (conservative):
  * signal s_t is computed from data up to the close of day t;
  * the order is filled at the close of day t+1 (EXEC_LAG = 1 day);
  * the first return earned is close(t+1) -> close(t+2).
  Hence the position applied to the return of day t is s_{t-(1+EXEC_LAG)}.

Costs: each unit of turnover pays (slippage ticks x tick value + commission) / contract
notional, where notional uses the previous close, so cost in bps varies with the price level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EXEC_LAG = 1

CONTRACTS = {
    # E-mini S&P 500: $50 x index, tick 0.25 = $12.50
    "ES": {"multiplier": 50.0, "tick_size": 0.25, "tick_value": 12.50},
    # E-mini Nasdaq-100: $20 x index, tick 0.25 = $5.00
    "NQ": {"multiplier": 20.0, "tick_size": 0.25, "tick_value": 5.00},
}
SLIPPAGE_TICKS_PER_SIDE = 1.0
COMMISSION_PER_SIDE_USD = 2.50  # broker + exchange + NFA fees, round number assumption


def lag_positions(signal: pd.Series, lag: int = EXEC_LAG) -> pd.Series:
    return signal.shift(1 + lag).fillna(0.0)


def cost_rate_per_side(close: pd.Series, root: str) -> pd.Series:
    c = CONTRACTS[root]
    usd = SLIPPAGE_TICKS_PER_SIDE * c["tick_value"] + COMMISSION_PER_SIDE_USD
    notional = close.shift(1) * c["multiplier"]
    return (usd / notional).bfill()


def net_returns(pos: pd.Series, ret: pd.Series, cost_rate: pd.Series) -> pd.Series:
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    return pos * ret - turnover * cost_rate


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve, with the starting capital 1.0 as the
    initial peak (equity = [1.0, cumprod(1 + r)]), so a loss on the first days is counted."""
    eq = np.concatenate([[1.0], (1.0 + r).cumprod().to_numpy(dtype=float)])
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def metrics(r: pd.Series, pos: pd.Series | None = None) -> dict:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {}
    eq = (1.0 + r).cumprod()
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else float("nan")
    sd = float(r.std(ddof=1))
    out = {
        "n_days": int(n),
        "cagr": cagr,
        "ann_vol": sd * np.sqrt(TRADING_DAYS),
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "max_drawdown": max_drawdown(r),
        "total_return": float(eq.iloc[-1] - 1.0),
    }
    if pos is not None:
        p = pos.reindex(r.index)
        out["pct_time_in_market"] = float((p != 0).mean())
        out["pct_time_long"] = float((p > 0).mean())
        out["pct_time_short"] = float((p < 0).mean())
        out["annual_turnover"] = float(p.diff().abs().fillna(0).sum() / years)
    return out


def sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def stationary_bootstrap_idx(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis and Romano (1994) stationary bootstrap indices."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    jumps = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, reps=2000, mean_block=20, seed=7) -> dict:
    """Paired stationary block bootstrap CI for Sharpe(a) - Sharpe(b)."""
    df = pd.concat([a, b], axis=1).dropna()
    x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        idx = stationary_bootstrap_idx(len(x), mean_block, rng)
        diffs[i] = sharpe(x[idx]) - sharpe(y[idx])
    return {
        "point": sharpe(x) - sharpe(y),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        # Share of bootstrap draws <= 0. The draws are centred on the point estimate, so this is
        # NOT a p-value (it is not computed under a null of zero difference); use the CI instead.
        "share_boot_draws_le_0": float((diffs <= 0).mean()),
        "reps": reps,
        "mean_block_days": mean_block,
    }


def run_lengths(labels: pd.Series) -> pd.DataFrame:
    """Contiguous episodes of the same label: label, start, end, n_days."""
    lab = labels.dropna()
    grp = (lab != lab.shift()).cumsum()
    ep = lab.groupby(grp).agg(["first", "size"])
    starts = lab.groupby(grp).apply(lambda s: s.index[0])
    ends = lab.groupby(grp).apply(lambda s: s.index[-1])
    return pd.DataFrame({"label": ep["first"].to_numpy(), "start": starts.to_numpy(),
                         "end": ends.to_numpy(), "n_days": ep["size"].to_numpy()})


def transition_matrix(labels: pd.Series, order: list[str]) -> pd.DataFrame:
    lab = labels.dropna()
    tm = pd.crosstab(lab.shift(1).iloc[1:], lab.iloc[1:])
    tm = tm.reindex(index=order, columns=order, fill_value=0)
    return tm.div(tm.sum(axis=1).replace(0, np.nan), axis=0)
