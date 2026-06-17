import os
import yaml
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from data.fetcher import DataFetcher
from data.features import FeatureEngineer
from env.trading_env import AutoTradingEnv
from models.ppo_lstm import make_ppo_lstm_policy
from utils.seeding import set_global_seed
from utils.logger import setup_logger
from utils.data_utils import split_train_val_test, fetch_all_synthetic

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=5, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0 and len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.no_improvement_count = 0
                self.model.save("best_model")
            else:
                self.no_improvement_count += 1
                if self.no_improvement_count >= self.patience:
                    print("Early stopping triggered.")
                    return False
        return True

def train_rl_agent(config_path: str = "config/config.yaml",
                   dry_run: bool = False,
                   total_timesteps: int = None):
    """Train the PPO agent on the in-sample *train* segment of the first symbol.

    Parameters
    ----------
    config_path     : path to the YAML config.
    dry_run         : when True, use deterministic synthetic data (no network)
                      and a small step budget so the pipeline can be smoke-tested.
    total_timesteps : optional override for ``training.total_timesteps``.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_global_seed(config['model']['seed'])
    logger = setup_logger("train")

    # Fetch and preprocess data
    if dry_run:
        logger.info("Dry-run mode: using synthetic offline data (no network).")
        raw_data = fetch_all_synthetic(config)
    else:
        fetcher = DataFetcher(config)
        raw_data = fetcher.fetch_all()
    engineer = FeatureEngineer(config)

    train_ratio = config['data'].get('train_ratio', 0.7)
    val_ratio = config['data'].get('val_ratio', 0.15)
    window = config['env']['window_size']

    all_envs = []
    for symbol, df in raw_data.items():
        logger.info(f"Processing {symbol}...")
        df = engineer.add_technical_indicators(df)
        df = engineer.normalize_features(df, fit_scaler=True)
        df = df.dropna().reset_index(drop=True)
        # Walk-forward: train only on the in-sample train segment, holding out
        # validation/test for backtest evaluation.
        train_df, _, _ = split_train_val_test(df, train_ratio, val_ratio)
        if len(train_df) <= window + 1:
            logger.warning(
                f"{symbol}: train segment too short; using full series.")
            train_df = df
        train_df = train_df.reset_index(drop=True)
        env = AutoTradingEnv(train_df, config, window, engineer.get_feature_dim(train_df))
        env = Monitor(env)
        all_envs.append(env)

    if not all_envs:
        raise RuntimeError("No usable symbols/environments to train on.")

    # Use first symbol for training
    train_env = DummyVecEnv([lambda: all_envs[0]])

    # Create model
    model = make_ppo_lstm_policy(train_env, config)

    # Callbacks
    eval_callback = EvalCallback(
        train_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=config['training']['eval_freq'],
        deterministic=True,
        render=False
    )

    early_stop_callback = EarlyStoppingCallback(patience=config['training']['early_stop_patience'])

    # Resolve the training budget: explicit override > dry-run smoke budget >
    # config value.
    if total_timesteps is not None:
        steps = int(total_timesteps)
    elif dry_run:
        steps = min(2048, int(config['training']['total_timesteps']))
    else:
        steps = int(config['training']['total_timesteps'])

    # Train
    logger.info(f"Starting training for {steps} timesteps...")
    model.learn(
        total_timesteps=steps,
        callback=[eval_callback, early_stop_callback],
        log_interval=config['training']['log_interval'],
        progress_bar=True
    )

    model.save("final_model")
    logger.info("Training completed. Model saved.")

if __name__ == "__main__":
    train_rl_agent()