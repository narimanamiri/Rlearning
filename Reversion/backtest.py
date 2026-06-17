import os
import yaml
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from data.fetcher import DataFetcher
from data.features import FeatureEngineer
from env.trading_env import AutoTradingEnv
from utils.seeding import set_global_seed
from utils.logger import setup_logger
from utils.reporting import (
    BacktestReport, export_combined_summary, plot_equity_curve)
from utils.data_utils import split_train_val_test, fetch_all_synthetic


def _periods_per_year(interval: str) -> int:
    """Annualization factor for Sharpe/Sortino based on the bar interval."""
    interval = (interval or "1d").lower()
    mapping = {
        "1d": 252, "1day": 252,
        "1h": 252 * 6, "60m": 252 * 6,
        "1wk": 52, "1week": 52,
        "1m": 252 * 390, "1min": 252 * 390,
    }
    return mapping.get(interval, 252)


def backtest_model(config_path: str = "config/config.yaml",
                   model_path: str = "best_model",
                   dry_run: bool = False,
                   report_dir: str = "reports",
                   export: bool = True,
                   plot: bool = False,
                   seed: int = None,
                   indicators=None):
    """Backtest a trained model on the held-out *test* segment of each symbol.

    Parameters
    ----------
    config_path : path to the YAML config.
    model_path  : path to the saved SB3 model (without ``.zip``).
    dry_run     : when True, use deterministic synthetic data instead of any
                  network fetch (no live calls). Still requires a saved model.
    report_dir  : directory for CSV/JSON report artifacts.
    export      : when True, write per-symbol equity/trade CSVs + JSON summary
                  and a combined cross-symbol summary.
    plot        : when True, also render a PNG equity-curve plot per symbol
                  (requires matplotlib; silently skipped if unavailable).
    seed        : optional override for ``model.seed`` (reproducibility).
    indicators  : optional override for ``data.indicators`` (feature set).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if seed is not None:
        config['model']['seed'] = int(seed)
    if indicators is not None:
        config.setdefault('data', {})['indicators'] = list(indicators)

    set_global_seed(config['model']['seed'])
    logger = setup_logger("backtest")

    if dry_run:
        logger.info("Dry-run mode: using synthetic offline data (no network).")
        raw_data = fetch_all_synthetic(config)
    else:
        fetcher = DataFetcher(config)
        raw_data = fetcher.fetch_all()

    engineer = FeatureEngineer(config)
    ppy = _periods_per_year(config['data'].get('interval', '1d'))
    train_ratio = config['data'].get('train_ratio', 0.7)
    val_ratio = config['data'].get('val_ratio', 0.15)
    initial_balance = config['env']['initial_balance']

    results = {}
    reports = {}
    for symbol, df in raw_data.items():
        logger.info(f"Backtesting on {symbol} (out-of-sample test segment)...")

        df = engineer.add_technical_indicators(df)
        df = df.dropna().reset_index(drop=True)

        # Walk-forward: evaluate only on the held-out test segment so results
        # reflect out-of-sample performance, not data the model trained on.
        train_df, _, test_df = split_train_val_test(df, train_ratio, val_ratio)
        window = config['env']['window_size']
        if len(test_df) <= window + 1:
            logger.warning(
                f"{symbol}: test segment too short ({len(test_df)} rows) for "
                f"window {window}; falling back to full series.")
            test_df = df
            train_df = df
        # Fit the scaler on this symbol's train segment and apply it to the test
        # segment so the model sees the *same normalized distribution* it was
        # trained on (previously the test features were left un-normalized,
        # silently breaking out-of-sample inference).
        engineer.fit_scalers(train_df)
        test_df = engineer.transform_features(test_df)
        test_df = test_df.reset_index(drop=True)

        env = AutoTradingEnv(test_df, config, window, engineer.get_feature_dim(test_df))
        model = PPO.load(model_path, env=env)

        obs, _ = env.reset()
        done = False
        info = {'net_worth': initial_balance}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        trade_history = env.get_trade_history()
        equity_curve = env.get_equity_curve()
        price_history = env.get_price_history()

        report = BacktestReport(
            symbol=symbol,
            equity_curve=equity_curve,
            trades=trade_history,
            metrics=None,  # computed from the equity curve inside the report
            prices=price_history,
        )
        # Recompute metrics with the proper annualization factor + initial balance.
        from utils.reporting import compute_metrics
        report.metrics = compute_metrics(
            equity_curve, trade_history,
            initial_balance=initial_balance,
            periods_per_year=ppy,
        )

        results[symbol] = report.metrics
        reports[symbol] = report

        if export:
            paths = report.export(report_dir)
            logger.info(f"{symbol}: wrote {', '.join(paths.values())}")

        if plot:
            png = plot_equity_curve(report, report_dir)
            if png:
                logger.info(f"{symbol}: wrote equity plot {png}")

        logger.info(f"{symbol} Backtest Results: {report.metrics}")

    if export and reports:
        combined = export_combined_summary(reports, report_dir)
        logger.info(f"Combined summary: {', '.join(combined.values())}")

    return results


if __name__ == "__main__":
    backtest_model()
