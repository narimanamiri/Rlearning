"""
Reversion pipeline entry point.

By default this trains a PPO agent on the in-sample (train) segment and then
backtests it out-of-sample, exporting CSV/JSON reports. A small CLI lets you
override the config path, run a fully-offline dry-run, restrict to a single
stage, and choose where reports go.

Examples
--------
    # Full run with the default config
    python main.py

    # Offline smoke test (synthetic data, no network, small step budget)
    python main.py --dry-run

    # Use a custom config and write reports elsewhere
    python main.py --config config/my_config.yaml --report-dir out/reports

    # Only backtest an already-trained model
    python main.py --backtest-only --model best_model
"""

import argparse

import yaml

from train import train_rl_agent
from evaluate import evaluate_performance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RL Auto-Trader (Reversion) — train and backtest a PPO agent.")
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to the YAML config file (default: config/config.yaml).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run end-to-end on synthetic offline data with a small step "
             "budget. No network calls. Useful for smoke-testing the pipeline.")
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train the model; skip backtest/evaluation.")
    parser.add_argument(
        "--backtest-only", action="store_true",
        help="Only backtest/evaluate an existing model; skip training.")
    parser.add_argument(
        "--model", default="best_model",
        help="Path (without .zip) to the saved model for backtesting "
             "(default: best_model).")
    parser.add_argument(
        "--report-dir", default=None,
        help="Directory for exported CSV/JSON backtest reports. Defaults to "
             "reporting.report_dir from the config (or 'reports').")
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override training.total_timesteps from the config.")
    parser.add_argument(
        "--no-export", action="store_true",
        help="Disable writing CSV/JSON report artifacts during backtest.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.train_only and args.backtest_only:
        raise SystemExit("--train-only and --backtest-only are mutually exclusive.")

    # Resolve report dir: CLI flag wins, else config's reporting.report_dir,
    # else "reports".
    report_dir = args.report_dir
    if report_dir is None:
        try:
            with open(args.config, "r") as fh:
                cfg = yaml.safe_load(fh)
            report_dir = (cfg.get("reporting", {}) or {}).get("report_dir", "reports")
        except (OSError, yaml.YAMLError):
            report_dir = "reports"

    print("Starting RL Auto-Trader (Reversion)...")
    if args.dry_run:
        print("[dry-run] synthetic offline data; no network calls.")

    if not args.backtest_only:
        train_rl_agent(
            config_path=args.config,
            dry_run=args.dry_run,
            total_timesteps=args.timesteps,
        )

    if not args.train_only:
        # evaluate_performance runs the backtest (which exports the CSV/JSON
        # reports) and then prints the aggregate summary, so we don't call
        # backtest_model separately here.
        evaluate_performance(
            config_path=args.config,
            model_path=args.model,
            dry_run=args.dry_run,
            report_dir=report_dir,
            export=not args.no_export,
        )

    print("Done.")


if __name__ == "__main__":
    main()
