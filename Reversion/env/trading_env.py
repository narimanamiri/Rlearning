import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class AutoTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, config: Dict, window_size: int, feature_dim: int):
        super(AutoTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.config = config
        self.window_size = window_size
        self.feature_dim = feature_dim

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = gym.spaces.Discrete(3)

        # Observation: (window_size, feature_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, feature_dim), dtype=np.float32
        )

        # Account state
        self.initial_balance = config['env']['initial_balance']
        self.transaction_cost_pct = config['env']['transaction_cost_pct']
        self.slippage_pct = config['env']['slippage_pct']
        self.max_position = config['env']['max_position']
        self.stop_loss_pct = config['env']['stop_loss_pct']
        self.risk_lambda = config['env']['risk_lambda']

        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trade_history = []
        self.position_history = []
        self.drawdown = 0

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        current_price = self.df.iloc[self.current_step]['close']
        info = {}

        # Execute action
        if action == 1:  # Buy
            self._buy(current_price)
        elif action == 2:  # Sell
            self._sell(current_price)
        # 0 = Hold → do nothing

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1 or self.net_worth <= 0.1 * self.initial_balance

        # Update net worth
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0

        # Reward calculation
        pnl = self.net_worth - self.prev_net_worth
        volatility = np.std([t['pnl'] for t in self.trade_history[-10:]]) if len(self.trade_history) >= 10 else 0.01
        risk_penalty = self.risk_lambda * (self.drawdown + volatility)
        reward = pnl - risk_penalty

        # Stop loss
        if self.drawdown > self.stop_loss_pct:
            self._liquidate(current_price)
            done = True

        truncated = False
        obs = self._get_observation() if not done else np.zeros((self.window_size, self.feature_dim), dtype=np.float32)

        info.update({
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'drawdown': self.drawdown,
            'step': self.current_step
        })

        return obs, reward, done, truncated, info

    def _buy(self, price):
        if self.shares_held >= self.max_position:
            return
        max_shares = self.balance / (price * (1 + self.transaction_cost_pct + self.slippage_pct))
        shares_bought = min(max_shares, self.max_position - self.shares_held)
        if shares_bought <= 0:
            return

        cost = shares_bought * price * (1 + self.transaction_cost_pct + self.slippage_pct)
        self.balance -= cost
        self.shares_held += shares_bought

        self.trade_history.append({
            'step': self.current_step,
            'type': 'buy',
            'price': price,
            'shares': shares_bought,
            'cost': cost,
            'pnl': 0
        })

    def _sell(self, price):
        if self.shares_held <= 0:
            return
        revenue = self.shares_held * price * (1 - self.transaction_cost_pct - self.slippage_pct)
        pnl = revenue - (self.shares_held * self.df.iloc[self.current_step - 1]['close'])  # approx cost basis
        self.balance += revenue
        self.trade_history.append({
            'step': self.current_step,
            'type': 'sell',
            'price': price,
            'shares': self.shares_held,
            'revenue': revenue,
            'pnl': pnl
        })
        self.shares_held = 0

    def _liquidate(self, price):
        if self.shares_held > 0:
            self._sell(price)

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        obs_df = self.df.iloc[start:end]
        feature_cols = [c for c in obs_df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        obs = obs_df[feature_cols].values.astype(np.float32)
        return obs

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: ${self.net_worth:.2f}, Drawdown: {self.drawdown:.2%}")

    def get_trade_history(self):
        return self.trade_history