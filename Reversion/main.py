"""
Reversion pipeline entry point.

By default this trains a PPO agent on the in-sample (train) segment and then
backtests it out-of-sample, exporting CSV/JSON reports. A small CLI lets you
override the config path, run a fully-offline dry-run, restrict to a single
stage, choose indicators, set a seed, plot equity curves, and build a
cross-run leaderboard.

Examples
--------
    # Full run with the default config
    python main.py

    # Offline smoke test (synthetic data, no network, small step budget)
    python main.py --dry-run --timesteps 2048

    # Reproducible run with extra indicators and an equity-curve PNG
    python main.py --dry-run --seed 7 --indicators base stochastic obv --plot

    # Use a custom config and write reports elsewhere
    python main.py --config config/my_config.yaml --report-dir out/reports

    # Only backtest an already-trained model
    python main.py --backtest-only --model best_model

    # Rank every past per-symbol result under the report dir (no training)
    python main.py --leaderboard --sort-by total_return_pct

Heavy dependencies (stable-baselines3 / gymnasium / torch) are imported lazily
inside ``main()`` *after* argument parsing, so ``python main.py --help`` and
``--leaderboard`` work even when those packages are not installed.
"""

import argparse

import yaml

# Indicator names are cheap to import (pure numpy/pandas module) and let
# --help advertise the valid choices.
from data.features import AVAILABLE_INDICATORS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "RL Auto-Trader (Reversion) - train a PPO agent on the in-sample\n"
            "segment and backtest it out-of-sample, exporting CSV/JSON reports."),
        epilog=(
            "examples:\n"
            "  python main.py --dry-run --timesteps 2048\n"
            "  python main.py --dry-run --seed 7 --indicators base obv --plot\n"
            "  python main.py --backtest-only --model best_model\n"
            "  python main.py --leaderboard --sort-by total_return_pct\n"))

    g_run = parser.add_argument_group("run mode")
    g_run.add_argument(
        "--dry-run", action="store_true",
        help="Run end-to-end on deterministic synthetic data with a small step "
             "budget. No network calls - ideal for smoke tests / CI.")
    g_run.add_argument(
        "--train-only", action="store_true",
        help="Only train the model; skip backtest/evaluation.")
    g_run.add_argument(
        "--backtest-only", action="store_true",
        help="Only backtest/evaluate an existing model; skip training.")
    g_run.add_argument(
        "--leaderboard", action="store_true",
        help="Don't train/backtest; aggregate every *_summary.json under the "
             "report dir into a ranked cross-run leaderboard and print it.")

    g_cfg = parser.add_argument_group("configuration")
    g_cfg.add_argument(
        "--config", default="config/config.yaml", metavar="PATH",
        help="Path to the YAML config file (default: config/config.yaml).")
    g_cfg.add_argument(
        "--model", default="best_model", metavar="PATH",
        help="Path (without .zip) to the saved model for backtesting "
             "(default: best_model).")
    g_cfg.add_argument(
        "--timesteps", type=int, default=None, metavar="N",
        help="Override training.total_timesteps from the config.")
    g_cfg.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help="Override model.seed for a reproducible run (wired through "
             "training, env reset, and backtest).")
    g_cfg.add_argument(
        "--indicators", nargs="+", default=None, metavar="NAME",
        choices=list(AVAILABLE_INDICATORS),
        help="Indicator set to engineer. Choices: "
             f"{', '.join(AVAILABLE_INDICATORS)}. "
             "'base' is always included (default: config value or 'base').")

    g_out = parser.add_argument_group("output")
    g_out.add_argument(
        "--report-dir", default=None, metavar="DIR",
        help="Directory for exported CSV/JSON backtest reports. Defaults to "
             "reporting.report_dir from the config (or 'reports').")
    g_out.add_argument(
        "--plot", action="store_true",
        help="Also render a PNG equity-curve plot per symbol during backtest "
             "(requires matplotlib; skipped with a hint if unavailable).")
    g_out.add_argument(
        "--no-export", action="store_true",
        help="Disable writing CSV/JSON report artifacts during backtest.")
    g_out.add_argument(
        "--sort-by", default="sharpe_ratio", metavar="METRIC",
        help="Metric to rank the --leaderboard by (default: sharpe_ratio).")
    g_out.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="With --leaderboard, show only the top-N rows.")
    return parser


def _resolve_report_dir(args) -> str:
    """CLI flag wins, else config's reporting.report_dir, else 'reports'."""
    if args.report_dir is not None:
        return args.report_dir
    try:
        with open(args.config, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        return (cfg.get("reporting", {}) or {}).get("report_dir", "reports")
    except (OSError, yaml.YAMLError):
        return "reports"


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Validate mutually-exclusive / conflicting stage flags up front with clear
    # messages rather than letting a half-run fail deep inside.
    exclusive = [n for n, v in (
        ("--train-only", args.train_only),
        ("--backtest-only", args.backtest_only),
        ("--leaderboard", args.leaderboard)) if v]
    if len(exclusive) > 1:
        raise SystemExit(
            f"{' and '.join(exclusive)} are mutually exclusive; pick one.")

    report_dir = _resolve_report_dir(args)

    # --leaderboard is a pure reporting command: no heavy deps, no network,
    # no model needed. Handle it before importing train/evaluate.
    if args.leaderboard:
        from utils.reporting import build_leaderboard
        df = build_leaderboard(report_dir, sort_by=args.sort_by, top=args.top)
        if df.empty:
            print(f"No *_summary.json reports found under '{report_dir}'. "
                  "Run a backtest first (e.g. `python main.py --dry-run`).")
            return
        print(f"\n=== LEADERBOARD (sorted by {args.sort_by}) ===")
        print(df.to_string(index=False))
        return

    print("Starting RL Auto-Trader (Reversion)...")
    if args.dry_run:
        print("[dry-run] synthetic offline data; no network calls.")
    if args.seed is not None:
        print(f"[seed] using seed {args.seed} for this run.")

    # Import the heavy pipeline modules lazily so --help / --leaderboard work
    # without stable-baselines3 / gymnasium / torch installed.
    try:
        from train import train_rl_agent
        from evaluate import evaluate_performance
    except ImportError as exc:
        raise SystemExit(
            f"Training/backtesting needs stable-baselines3, gymnasium and "
            f"torch, which failed to import:\n    {exc}\n"
            f"Install them with `pip install -r requirements.txt` "
            f"(or use --leaderboard, which doesn't need them).")

    if not args.backtest_only:
        train_rl_agent(
            config_path=args.config,
            dry_run=args.dry_run,
            total_timesteps=args.timesteps,
            seed=args.seed,
            indicators=args.indicators,
        )

    if not args.train_only:
        # evaluate_performance runs the backtest (which exports the CSV/JSON
        # reports) and then prints the aggregate summary.
        evaluate_performance(
            config_path=args.config,
            model_path=args.model,
            dry_run=args.dry_run,
            report_dir=report_dir,
            export=not args.no_export,
            plot=args.plot,
            seed=args.seed,
            indicators=args.indicators,
        )

    print("Done.")


if __name__ == "__main__":
    main()
