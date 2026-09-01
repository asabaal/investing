"""Indicator library - pure pandas, no vendor calls (plan §5 feature engine)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def ema(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # Zero-loss with real gains (monotonic up) -> RSI 100 convention.
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    return out


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std()
    return mid - num_std * sd, mid, mid + num_std * sd


def MACD(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def realized_vol(returns: pd.Series, window: int = 20, annualize: int | None = 252) -> pd.Series:
    rv = returns.rolling(window, min_periods=window).std()
    if annualize:
        rv = rv * np.sqrt(annualize)
    return rv


def dollar_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return (df["close"] * df["volume"]).rolling(window, min_periods=1).mean()


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0
