# Rlearning — Reinforcement-Learning Trading Experiments

This repository contains two independent reinforcement-learning trading
projects built on [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
and [Gymnasium](https://gymnasium.farama.org/):

1. **`app.py`** — a monolithic "Adaptive Crypto Trading System" with optional
   MetaTrader 5 integration, an adaptive reward/exploration scheme, and a
   simulation/live runner.
2. **`Reversion/`** — a smaller, modular PPO project (data → features → env →
   train → backtest) driven by a YAML config.

> Research / educational use only — not financial advice. The trading logic is
> illustrative and is **not** intended for use with real funds.

---

## 1. `app.py` — Adaptive Crypto Trading System

### Components
- **`DataAcquirer`** — pulls 1-minute OHLCV from MetaTrader 5
  (`copy_rates_from_pos`) and falls back to Binance/CCXT or synthetic data when
  MT5 is unavailable; also exposes real-time MT5 ticks.
- **`FeatureEngineer`** — returns, SMA/EMA, RSI, and MACD indicators.
- **`CryptoTradingEnvironment`** — a Gymnasium env with a discrete action space
  (`0=Hold, 1=Buy, 2=Sell, 3=Close`) and a flat 1-D observation made of a
  look-back window of price/indicator values plus portfolio state. The reward is
  performance-adaptive and includes overtrading and risk penalties.
- **`AdaptiveTradingAgent`** — wraps an SB3 `PPO` ("MlpPolicy") model with
  adaptive exploration and feedback-driven retraining.
- **`CryptoTradingSystem`** — coordinates multiple agents, training, simulation
  and (optional) live MT5 order execution, and performance reporting.

### Run
```bash
python app.py
```
This initializes agents for a couple of symbols, trains them for a reduced
number of steps, runs a short simulation in a background thread, and prints a
JSON performance report. Tune everything in the `Config` class at the top of
`app.py` (symbols, training steps, indicator windows, file paths, etc.).

> MetaTrader 5 and `MetaTrader5`/`pytesseract`/`easyocr`/`selenium` are imported
> at module load. On platforms without MT5 the acquirer logs the failure and
> falls back to CCXT/synthetic data, but the third-party packages still need to
> be importable.

## 2. `Reversion/` — Modular PPO Project

A cleaner, config-driven pipeline. Entry point is `Reversion/main.py`, which
trains a PPO agent and then backtests it.

### Layout
- `data/fetcher.py` — OHLCV from yfinance (stocks/`*-USD`) or ccxt/Binance
  (crypto); columns are normalized to lowercase `open/high/low/close/volume`.
- `data/features.py` — technical indicators + per-feature `StandardScaler`.
- `env/trading_env.py` — `AutoTradingEnv` with a `(window_size, feature_dim)`
  observation and a `0=Hold, 1=Buy, 2=Sell` action space.
- `models/ppo_lstm.py` — PPO with a custom LSTM features extractor over the
  look-back window (standard `MlpPolicy`; the recurrence lives in the
  extractor).
- `train.py`, `backtest.py`, `evaluate.py` — training, backtesting, evaluation.
- `utils/reporting.py` — performance metrics + CSV/JSON report exporter.
- `utils/data_utils.py` — walk-forward train/val/test split + synthetic data.
- `config/config.yaml` — all hyperparameters and data settings.

### Run
```bash
cd Reversion
python main.py
```
The modules use package-relative imports (`from data.fetcher import ...`), so
run them from inside the `Reversion/` directory. Adjust symbols, window size,
and PPO hyperparameters in `config/config.yaml`.

### CLI options (new)
`main.py` now has a small command-line interface:

`main.py --help` works **without** `stable-baselines3`/`gymnasium`/`torch`
installed — the heavy training/backtest imports are deferred until they are
actually needed, so `--help` and `--leaderboard` run anywhere.

```bash
python main.py --help
python main.py --config config/config.yaml       # override the config path
python main.py --dry-run                          # offline smoke test, no network
python main.py --train-only                       # train, skip backtest/eval
python main.py --backtest-only --model best_model # evaluate an existing model
python main.py --report-dir out/reports           # where reports are written
python main.py --timesteps 5000                   # override training budget
python main.py --no-export                         # skip CSV/JSON report files
python main.py --seed 7                            # reproducible run
python main.py --indicators base stochastic obv    # choose the feature set
python main.py --plot                              # export equity-curve PNGs
python main.py --leaderboard --sort-by total_return_pct  # rank past runs
```

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to the YAML config (default `config/config.yaml`). |
| `--dry-run` | Run end-to-end on **deterministic synthetic data** with a small step budget. No network calls. |
| `--train-only` / `--backtest-only` / `--leaderboard` | Restrict to a single stage (mutually exclusive). |
| `--model PATH` | Saved model path (without `.zip`) for backtesting (default `best_model`). |
| `--report-dir DIR` | Output dir for reports (defaults to `reporting.report_dir` in the config, else `reports`). |
| `--timesteps N` | Override `training.total_timesteps`. |
| `--seed N` | Override `model.seed`; wired through training, env reset and backtest for reproducibility. |
| `--indicators NAME [NAME ...]` | Indicator groups to engineer (`base stochastic obv roc williams_r cci`); `base` is always kept. |
| `--plot` | Render a PNG equity-curve plot per symbol during backtest (matplotlib; skipped with a hint if unavailable). |
| `--no-export` | Disable writing CSV/JSON report artifacts. |
| `--sort-by METRIC` / `--top N` | Ranking metric and row cap for `--leaderboard`. |

### New features

**1. Walk-forward train/test split.** `data.train_ratio` / `data.val_ratio`
in `config.yaml` are now actually used: each symbol's series is split
chronologically into train / validation / test. Training runs on the
**in-sample train** segment and the backtest runs on the **held-out test**
segment, so reported numbers reflect out-of-sample performance. Implemented in
`utils/data_utils.py::split_train_val_test` (with input validation and a
short-segment fallback).

**2. Backtest report exporter (CSV + JSON + summary stats).** After a backtest,
`utils/reporting.py` computes total return, CAGR, annualized Sharpe & Sortino,
max drawdown, volatility, win rate, profit factor, and trade count, then writes
per-symbol artifacts plus a combined cross-symbol summary into `--report-dir`:

```
reports/BTC-USD_<ts>_equity.csv     # step, net_worth, price  (equity curve)
reports/BTC-USD_<ts>_trades.csv     # full trade log
reports/BTC-USD_<ts>_summary.json   # metrics for that symbol
reports/backtest_summary_<ts>.csv   # one row per symbol
reports/backtest_summary_<ts>.json
```

The metric functions (`compute_metrics`, `max_drawdown`, `sharpe_ratio`,
`sortino_ratio`) depend only on `numpy`/`pandas` and can be imported and tested
without `stable-baselines3`/`torch`.

**3. Equity-curve recording.** `AutoTradingEnv` now records a per-step net-worth
series and aligned close-price series, exposed via `get_equity_curve()` /
`get_price_history()` and exported as the `*_equity.csv` above — ready to plot.

**4. Paper / dry-run mode.** `--dry-run` swaps the live data fetch for a
deterministic synthetic OHLCV generator (`utils/data_utils.py`) and uses a
reduced training budget, so the **entire pipeline** (data → features → env →
train → backtest → report) runs end-to-end with **no network access**. Great
for CI / smoke-testing changes:

```bash
cd Reversion
python main.py --dry-run --timesteps 2048
```

**5. Selectable indicator sets.** `data.indicators` in `config.yaml` (or the
`--indicators` flag) chooses which technical-indicator groups
`FeatureEngineer` builds. `base` (log return, SMAs, RSI, MACD, Bollinger, ATR,
volatility) is always included; opt-in extras are `stochastic` (%K/%D), `obv`
(on-balance volume), `roc` (rate of change / momentum), `williams_r`
(Williams %R) and `cci` (commodity channel index). The observation
`feature_dim` adapts automatically, and unknown names are rejected with a clear
error.

```bash
python main.py --dry-run --indicators base stochastic obv cci
```

**6. Reproducible runs (`--seed`).** `--seed N` overrides `model.seed` and is
threaded through global seeding, the synthetic-data generator, the env reset,
and PPO, so two runs with the same seed produce identical results.

```bash
python main.py --dry-run --seed 7
```

**7. Equity-curve plot export (`--plot`).** During backtest, `--plot` renders a
PNG per symbol (net-worth line with an optional price overlay and the total
return in the title) next to the CSV/JSON reports, e.g.
`reports/BTC-USD_<ts>_equity.png`. `matplotlib` is imported lazily and guarded:
if it is missing the run prints a hint and continues rather than crashing.

```bash
python main.py --dry-run --plot
```

**8. Cross-run metrics leaderboard (`--leaderboard`).** Aggregates every
per-symbol `*_summary.json` under the report dir into one ranked table (one row
per symbol per run, with its timestamp), writes `leaderboard_<ts>.csv`, and
prints it. Pure `pandas` — no model, network or heavy deps required, so it runs
even where `stable-baselines3` is not installed.

```bash
python main.py --leaderboard --sort-by total_return_pct --top 10
```

### Correctness fixes in this pass
- **Out-of-sample normalization.** The backtest previously created a fresh
  `FeatureEngineer` and called `normalize_features(fit_scaler=False)` with an
  empty scaler, so **test features were never scaled** while training used
  scaled features — a silent train/serve skew. Scalers are now fit on each
  symbol's **train** segment and applied to its **test** segment (in both
  `train.py` and `backtest.py`), which also removes the prior look-ahead leak.
- **Realized-PnL cost basis.** `AutoTradingEnv._sell` now measures PnL against
  the actual (volume-weighted) entry price of the held shares instead of the
  previous bar's close, so trade PnL and win-rate/profit-factor are accurate.
- **Indicator math.** MACD now uses the standard `adjust=False` recursive EMA,
  and RSI is guarded against divide-by-zero (no-loss windows → 100).

---

## Requirements
- Python 3.10+
- Core: `stable-baselines3`, `gymnasium`, `torch`, `pandas`, `numpy`,
  `scikit-learn`, `matplotlib`, `pyyaml`
- Data: `ccxt`, `yfinance` (and `MetaTrader5` for `app.py`'s live/MT5 path)

Install the full set with:
```bash
pip install -r requirements.txt
```
`requirements.txt` is intentionally broad (it covers optional OCR/scraping/DB
extras imported by `app.py`). The `Reversion/` project only needs the core RL
and data packages listed above (`Reversion/requirements.txt`).

## Generated output
Models, charts, logs, and data are written to `models/`, `charts/`, `logs/`,
and `data/` and are git-ignored.
