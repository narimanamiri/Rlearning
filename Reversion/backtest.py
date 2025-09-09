import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from data.fetcher import DataFetcher
from data.features import FeatureEngineer
from env.trading_env import AutoTradingEnv
from utils.seeding import set_global_seed
from utils.logger import setup_logger

def backtest_model(config_path: str = "config/config.yaml", model_path: str = "best_model"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_global_seed(config['model']['seed'])
    logger = setup_logger("backtest")

    fetcher = DataFetcher(config)
    raw_data = fetcher.fetch_all()
    engineer = FeatureEngineer(config)

    results = {}
    for symbol, df in raw_data.items():
        logger.info(f"Backtesting on {symbol}...")

        df = engineer.add_technical_indicators(df)
        df = engineer.normalize_features(df, fit_scaler=False)

        env = AutoTradingEnv(df, config, config['env']['window_size'], engineer.get_feature_dim(df))
        model = PPO.load(model_path, env=env)

        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        trade_history = env.get_trade_history()
        net_worth = info['net_worth']
        returns = (net_worth - config['env']['initial_balance']) / config['env']['initial_balance']

        # Calculate metrics
        pnl_list = [t['pnl'] for t in trade_history if 'pnl' in t and t['pnl'] != 0]
        if len(pnl_list) > 0:
            sharpe = np.mean(pnl_list) / (np.std(pnl_list) + 1e-8) * np.sqrt(252)
            sortino = np.mean(pnl_list) / (np.std([x for x in pnl_list if x < 0]) + 1e-8) * np.sqrt(252)
        else:
            sharpe = sortino = 0

        max_drawdown = max([t.get('drawdown', 0) for t in trade_history]) if trade_history else 0

        results[symbol] = {
            'final_net_worth': net_worth,
            'total_return_pct': returns * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': len(trade_history),
            'win_rate': np.mean([t['pnl'] > 0 for t in trade_history if 'pnl' in t]) * 100 if pnl_list else 0
        }

        logger.info(f"{symbol} Backtest Results: {results[symbol]}")

    return results