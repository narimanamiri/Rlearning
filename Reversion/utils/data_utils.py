"""
Data-splitting and offline-data helpers for the Reversion pipeline.

Provides:
  * ``split_train_val_test`` — chronological (walk-forward style) split of a
    single OHLCV+features DataFrame into train / validation / test segments
    using the ``train_ratio`` / ``val_ratio`` from the config.
  * ``generate_synthetic_ohlcv`` — deterministic synthetic OHLCV used by the
    pipeline's ``--dry-run`` mode so the end-to-end flow can be exercised
    without any network calls.

Depends only on ``numpy`` / ``pandas`` (core deps).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def split_train_val_test(df: pd.DataFrame,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split ``df`` into (train, val, test) segments.

    The split is purely sequential (no shuffling) to preserve time ordering,
    which is required for any walk-forward / out-of-sample backtest. The test
    segment receives whatever remains after train and val.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1 to leave a test set")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def generate_synthetic_ohlcv(symbol: str,
                             n: int = 800,
                             interval: str = "1d",
                             seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for offline / dry-run use.

    Produces a geometric-random-walk close series with realistic-looking
    OHLC and volume columns. Deterministic given ``symbol`` + ``seed`` so
    repeated dry-runs are reproducible.
    """
    # Derive a stable per-symbol seed so different symbols look different but
    # each is reproducible.
    rng = np.random.default_rng(seed + (abs(hash(symbol)) % 10_000))

    base_price = 30000.0 if "BTC" in symbol.upper() else (
        2000.0 if "ETH" in symbol.upper() else 150.0)
    daily_vol = 0.02  # 2% per step

    shocks = rng.normal(0.0, daily_vol, size=n)
    # Mild mean-reverting drift toward the base price.
    closes = np.empty(n, dtype=float)
    price = base_price
    for i in range(n):
        price *= (1.0 + shocks[i])
        price = price * 0.999 + base_price * 0.001  # gentle mean reversion
        closes[i] = max(price, 1e-6)

    opens = closes * (1.0 + rng.normal(0.0, daily_vol / 4, size=n))
    highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 3, size=n)))
    lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 3, size=n)))
    volumes = rng.lognormal(mean=10.0, sigma=1.0, size=n)

    freq = "1H" if interval in ("1h", "1H", "60m") else "1D"
    start = datetime.now() - timedelta(days=n)
    index = pd.date_range(start=start, periods=n, freq=freq)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=index)


def fetch_all_synthetic(config: Dict) -> Dict[str, pd.DataFrame]:
    """Build a {symbol: synthetic OHLCV} dict mirroring ``DataFetcher.fetch_all``.

    Used by the dry-run / paper mode so the pipeline runs end-to-end with no
    network access.
    """
    seed = config.get("model", {}).get("seed", 42)
    interval = config.get("data", {}).get("interval", "1d")
    out: Dict[str, pd.DataFrame] = {}
    for symbol in config["data"]["symbols"]:
        out[symbol] = generate_synthetic_ohlcv(symbol, interval=interval, seed=seed)
        print(f"[dry-run] generated synthetic data for {symbol} "
              f"({len(out[symbol])} bars)")
    return out
