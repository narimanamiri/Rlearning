"""
Backtest reporting utilities.

Computes summary performance statistics from an equity curve (and optional
trade log) and exports them to CSV / JSON. Deliberately depends only on
``numpy`` and ``pandas`` (both core deps) so it can be imported and unit-tested
without ``stable_baselines3``/``torch``/network access.

Typical usage::

    from utils.reporting import compute_metrics, BacktestReport

    metrics = compute_metrics(equity_curve, trades, initial_balance=10000)
    report = BacktestReport("BTC-USD", equity_curve, trades, metrics)
    report.export("reports")   # writes reports/BTC-USD_*.csv / .json
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _to_array(equity_curve: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(equity_curve), dtype=float)
    return arr[np.isfinite(arr)]


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Return the maximum drawdown as a positive fraction (0.2 == 20%)."""
    arr = _to_array(equity_curve)
    if arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    # Guard against zero / negative running max.
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_max > 0, (running_max - arr) / running_max, 0.0)
    return float(np.nanmax(drawdowns)) if drawdowns.size else 0.0


def sharpe_ratio(returns: Sequence[float], periods_per_year: int = 252,
                 risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from a series of per-period returns."""
    arr = _to_array(returns)
    if arr.size < 2:
        return 0.0
    excess = arr - (risk_free_rate / periods_per_year)
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: Sequence[float], periods_per_year: int = 252,
                  risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (downside-deviation denominator)."""
    arr = _to_array(returns)
    if arr.size < 2:
        return 0.0
    excess = arr - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0
    downside_std = np.std(downside, ddof=1) if downside.size > 1 else np.std(downside)
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def compute_metrics(equity_curve: Sequence[float],
                    trades: Optional[List[Dict]] = None,
                    initial_balance: Optional[float] = None,
                    periods_per_year: int = 252,
                    risk_free_rate: float = 0.0) -> Dict[str, float]:
    """Compute a dictionary of summary statistics from an equity curve.

    Parameters
    ----------
    equity_curve : sequence of net-worth values, one per step.
    trades       : optional list of trade dicts (expects a ``pnl`` key on
                   closing trades). Used for win-rate / profit-factor.
    initial_balance : starting capital; defaults to ``equity_curve[0]``.
    """
    arr = _to_array(equity_curve)
    if arr.size == 0:
        return {
            "initial_balance": float(initial_balance or 0.0),
            "final_net_worth": float(initial_balance or 0.0),
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility_pct": 0.0,
            "num_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "num_steps": 0,
        }

    start = float(initial_balance) if initial_balance is not None else float(arr[0])
    final = float(arr[-1])
    total_return = (final - start) / start if start != 0 else 0.0

    # Per-step simple returns for Sharpe/Sortino/volatility.
    step_returns = np.diff(arr) / arr[:-1] if arr.size > 1 else np.array([])
    step_returns = step_returns[np.isfinite(step_returns)]

    n_periods = max(arr.size - 1, 1)
    years = n_periods / periods_per_year
    if years > 0 and start > 0 and final > 0:
        cagr = (final / start) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    vol = float(np.std(step_returns, ddof=1) * np.sqrt(periods_per_year)) if step_returns.size > 1 else 0.0

    # Trade-based stats.
    pnls = []
    if trades:
        pnls = [t["pnl"] for t in trades if isinstance(t, dict)
                and "pnl" in t and t["pnl"] != 0]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = (len(wins) / len(pnls) * 100.0) if pnls else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)

    return {
        "initial_balance": round(start, 4),
        "final_net_worth": round(final, 4),
        "total_return_pct": round(total_return * 100.0, 4),
        "cagr_pct": round(cagr * 100.0, 4),
        "sharpe_ratio": round(sharpe_ratio(step_returns, periods_per_year, risk_free_rate), 4),
        "sortino_ratio": round(sortino_ratio(step_returns, periods_per_year, risk_free_rate), 4),
        "max_drawdown_pct": round(max_drawdown(arr) * 100.0, 4),
        "volatility_pct": round(vol * 100.0, 4),
        "num_trades": len(trades) if trades else 0,
        "win_rate_pct": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else profit_factor,
        "num_steps": int(arr.size),
    }


class BacktestReport:
    """Bundles an equity curve, trade log and metrics for one symbol/run."""

    def __init__(self, symbol: str,
                 equity_curve: Sequence[float],
                 trades: Optional[List[Dict]] = None,
                 metrics: Optional[Dict] = None,
                 prices: Optional[Sequence[float]] = None):
        self.symbol = symbol
        self.equity_curve = list(equity_curve)
        self.trades = list(trades) if trades else []
        self.metrics = metrics if metrics is not None else compute_metrics(
            equity_curve, trades)
        self.prices = list(prices) if prices is not None else None
        self.created_at = datetime.now().isoformat(timespec="seconds")

    def _safe_symbol(self) -> str:
        return str(self.symbol).replace("/", "_").replace("\\", "_")

    def equity_dataframe(self) -> pd.DataFrame:
        data = {"step": list(range(len(self.equity_curve))),
                "net_worth": self.equity_curve}
        if self.prices is not None and len(self.prices) == len(self.equity_curve):
            data["price"] = self.prices
        return pd.DataFrame(data)

    def export(self, out_dir: str = "reports", prefix: str = "") -> Dict[str, str]:
        """Write equity-curve CSV, trades CSV and a JSON summary.

        Returns a dict of {artifact: path}.
        """
        os.makedirs(out_dir, exist_ok=True)
        sym = self._safe_symbol()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{prefix}{sym}_{stamp}" if prefix else f"{sym}_{stamp}"
        paths: Dict[str, str] = {}

        equity_path = os.path.join(out_dir, f"{base}_equity.csv")
        self.equity_dataframe().to_csv(equity_path, index=False)
        paths["equity_csv"] = equity_path

        if self.trades:
            trades_path = os.path.join(out_dir, f"{base}_trades.csv")
            pd.DataFrame(self.trades).to_csv(trades_path, index=False)
            paths["trades_csv"] = trades_path

        summary_path = os.path.join(out_dir, f"{base}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump({
                "symbol": self.symbol,
                "created_at": self.created_at,
                "metrics": self.metrics,
            }, fh, indent=2, default=str)
        paths["summary_json"] = summary_path

        return paths


def export_combined_summary(reports: Dict[str, "BacktestReport"],
                            out_dir: str = "reports",
                            filename: str = "backtest_summary") -> Dict[str, str]:
    """Write a combined multi-symbol summary (CSV + JSON) across reports."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = {sym: rep.metrics for sym, rep in reports.items()}

    paths: Dict[str, str] = {}
    if rows:
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "symbol"
        csv_path = os.path.join(out_dir, f"{filename}_{stamp}.csv")
        df.to_csv(csv_path)
        paths["summary_csv"] = csv_path

    json_path = os.path.join(out_dir, f"{filename}_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)
    paths["summary_json"] = json_path
    return paths
