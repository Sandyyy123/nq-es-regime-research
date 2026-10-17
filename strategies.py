"""Strategy sleeves. Each signal at date t uses features known at the close of t only.

Signals are target positions in {-1, 0, +1} (one contract-notional unit). The execution
lag is applied later in backtest.py, never here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def trend_signal(feat: pd.DataFrame) -> pd.Series:
    """Trend following: long when 50d MA > 200d MA, short otherwise (flat until both exist)."""
    s = np.where(feat["ma50"] > feat["ma200"], 1.0, -1.0)
    s = pd.Series(s, index=feat.index)
    s[feat["ma200"].isna()] = 0.0
    return s


def meanrev_signal(feat: pd.DataFrame, z_entry: float = 1.0) -> pd.Series:
    """Short-term mean reversion on a volatility-scaled 5-day return z-score.

    z = r5 / (daily vol * sqrt(5)), daily vol from the trailing 20d realised vol.
    Long when z < -z_entry, short when z > +z_entry, flat otherwise.
    """
    daily_vol = feat["rv20"] / np.sqrt(TRADING_DAYS)
    z = feat["r5"] / (daily_vol * np.sqrt(5))
    s = np.where(z < -z_entry, 1.0, np.where(z > z_entry, -1.0, 0.0))
    s = pd.Series(s, index=feat.index)
    s[z.isna()] = 0.0
    return s


def long_signal(feat: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=feat.index)


def flat_signal(feat: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=feat.index)


SLEEVES = {"trend": trend_signal, "meanrev": meanrev_signal}
