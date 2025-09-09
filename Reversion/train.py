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

def train_rl_agent(config_path: str = "config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_global_seed(config['model']['seed'])
    logger = setup_logger("train")

    # Fetch and preprocess data
    fetcher = DataFetcher(config)
    raw_data = fetcher.fetch_all()
    engineer = FeatureEngineer(config)

    all_envs = []
    for symbol, df in raw_data.items():
        logger.info(f"Processing {symbol}...")
        df = engineer.add_technical_indicators(df)
        df = engineer.normalize_features(df, fit_scaler=True)
        env = AutoTradingEnv(df, config, config['env']['window_size'], engineer.get_feature_dim(df))
        env = Monitor(env)
        all_envs.append(env)

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

    # Train
    logger.info("Starting training...")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=[eval_callback, early_stop_callback],
        log_interval=config['training']['log_interval'],
        progress_bar=True
    )

    model.save("final_model")
    logger.info("Training completed. Model saved.")

if __name__ == "__main__":
    train_rl_agent()