import os
import time
import random
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
import warnings
import re
import json
from urllib.parse import urljoin
import csv
import io
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class RealisticDataGenerator:
    """
    Generates realistic forex data with proper trends, volatility, and patterns
    """
    
    def __init__(self):
        self.base_prices = {
            'EUR/USD': 1.08,
            'GBP/USD': 1.26,
            'USD/JPY': 151.50,
            'USD/CHF': 0.88,
            'USD/CAD': 1.36,
            'AUD/USD': 0.66,
            'NZD/USD': 0.61
        }
        
    def generate_realistic_data(self, symbol, num_points=500, timeframe='H1'):
        """Generate realistic forex data with trends and patterns"""
        # Determine time frequency based on timeframe
        if timeframe == 'H1':
            freq = 'H'
        elif timeframe == 'D1':
            freq = 'D'
        else:
            freq = '10T'  # Default to 10 minutes
            
        dates = pd.date_range(end=datetime.now(), periods=num_points, freq=freq)
        base_price = self.base_prices.get(symbol, 1.0)
        
        # Different volatility for different pairs
        volatility_map = {
            'EUR/USD': 0.0008,
            'GBP/USD': 0.0010,
            'USD/JPY': 0.0012,
            'USD/CHF': 0.0009,
            'USD/CAD': 0.0007,
            'AUD/USD': 0.0011,
            'NZD/USD': 0.0013
        }
        
        volatility = volatility_map.get(symbol, 0.0010)
        
        prices = []
        current_price = base_price
        
        # Generate realistic price movements with trends and reversals
        trend_direction = random.choice([-1, 1]) * random.uniform(0.0002, 0.0005)
        trend_duration = random.randint(50, 150)  # How long the trend lasts
        trend_counter = 0
        
        for i in range(num_points):
            # Change trend occasionally
            if trend_counter >= trend_duration:
                trend_direction = random.choice([-1, 1]) * random.uniform(0.0002, 0.0005)
                trend_duration = random.randint(50, 150)
                trend_counter = 0
            
            # Add trend component
            current_price += trend_direction * current_price
            
            # Add random noise with volatility
            change = random.normalvariate(0, volatility) * current_price
            current_price += change
            
            # Add occasional larger moves (news events)
            if random.random() < 0.02:  # 2% chance of a news event
                news_impact = random.normalvariate(0, volatility * 3) * current_price
                current_price += news_impact
            
            trend_counter += 1
            prices.append(current_price)
        
        # Create OHLC data
        opens = []
        highs = []
        lows = []
        closes = []
        
        for i, price in enumerate(prices):
            if i == 0:
                open_price = base_price
            else:
                open_price = closes[i-1] * (1 + random.uniform(-0.0001, 0.0001))
            
            close_price = price
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.0005))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.0005))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
        
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': [random.randint(1000000, 5000000) for _ in range(num_points)]
        }, index=dates)
        
        return df

class EnhancedFeatureEngineer:
    """Enhanced feature engineering with more technical indicators"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        # Price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['Close']  # ATR as percentage of price
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Volatility
        df['volatility_20'] = df['Close'].rolling(window=20).std()
        
        # Price rate of change
        df['roc_5'] = df['Close'].pct_change(5)
        df['roc_10'] = df['Close'].pct_change(10)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def normalize_features(self, df, feature_columns):
        """Normalize features using robust scaling"""
        for column in feature_columns:
            if column in df.columns:
                median = df[column].median()
                iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
                if iqr > 0:
                    df[column] = (df[column] - median) / iqr
        return df
    
    def prepare_features(self, df):
        """Prepare all features for the RL model"""
        df = self.calculate_technical_indicators(df)
        
        feature_columns = [
            'returns', 'log_returns', 'sma_10', 'sma_20', 'sma_50', 
            'macd', 'macd_signal', 'macd_hist', 'rsi', 'bb_width', 
            'bb_position', 'atr', 'atr_pct', 'volume_ratio', 'momentum_5',
            'momentum_10', 'volatility_20', 'roc_5', 'roc_10'
        ]
        
        # Keep only columns that exist
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        df = self.normalize_features(df, feature_columns)
        
        # Create windowed observations
        observations = []
        for i in range(len(df) - self.window_size + 1):
            obs = df[feature_columns].iloc[i:i+self.window_size].values
            observations.append(obs)
        
        return observations, df.index[self.window_size - 1:]

class ImprovedAutoTradingEnv(gym.Env):
    """Improved Trading Environment with better reward function"""
    
    def __init__(self, data, initial_balance=10000, transaction_cost=0.0005, risk_free_rate=0.02/252):
        super(ImprovedAutoTradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # Reduced transaction cost
        self.risk_free_rate = risk_free_rate
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observations: window_size x feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(data[0].shape[0], data[0].shape[1]), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = 0
        self.current_step = 0
        self.done = False
        self.portfolio_history = [self.initial_balance]
        self.position = 0  # 0: no position, 1: long, -1: short
        
        return self.data[self.current_step]
    
    def step(self, action):
        current_price = 1.0  # Using 1.0 as reference price for forex
        prev_net_worth = self.net_worth
        
        # Execute action
        if action == 1 and self.position != 1:  # Buy
            # Close short position if exists
            if self.position == -1:
                self.balance = self.holdings * current_price * (1 - self.transaction_cost)
                self.holdings = 0
                self.trades += 1
            
            # Open long position
            if self.balance > 0:
                self.holdings = self.balance / current_price
                self.balance = 0
                self.trades += 1
                self.position = 1
                
        elif action == 2 and self.position != -1:  # Sell
            # Close long position if exists
            if self.position == 1:
                self.balance = self.holdings * current_price * (1 - self.transaction_cost)
                self.holdings = 0
                self.trades += 1
            
            # Open short position
            if self.balance > 0:
                self.holdings = self.balance / current_price
                self.balance = 0
                self.trades += 1
                self.position = -1
                
        elif action == 0:  # Hold
            # Close position if we're in one and want to hold (more conservative)
            if self.position != 0:
                self.balance = self.holdings * current_price * (1 - self.transaction_cost)
                self.holdings = 0
                self.trades += 1
                self.position = 0
        
        # Update net worth
        if self.position == 1:  # Long
            self.net_worth = self.holdings * current_price
        elif self.position == -1:  # Short
            self.net_worth = self.balance + self.holdings * (2 - current_price)  # Profit when price goes down
        else:  # No position
            self.net_worth = self.balance
            
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Calculate reward - more focused on profitability
        portfolio_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        
        # Reduced penalty for trading
        trade_penalty = 0.0001 * self.trades
        
        # Risk-adjusted reward (simplified Sharpe ratio)
        if len(self.portfolio_history) > 1:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            risk_penalty = 0.05 * np.std(returns) if len(returns) > 0 else 0
        else:
            risk_penalty = 0
        
        # Drawdown penalty
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        drawdown_penalty = 0.1 * drawdown
        
        reward = portfolio_return - trade_penalty - risk_penalty - drawdown_penalty
        
        # Update step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Record portfolio value
        self.portfolio_history.append(self.net_worth)
        
        # Get next observation
        obs = self.data[self.current_step] if not self.done else self.data[-1]
        
        return obs, reward, self.done, {}
    
    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        position = "Long" if self.position == 1 else "Short" if self.position == -1 else "Neutral"
        print(f'Step: {self.current_step}, Position: {position}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}')

# PolicyNetwork and PPOAgent remain largely the same but with adjusted hyperparameters

class ImprovedPPOAgent:
    """Improved PPO Agent with better exploration"""
    
    def __init__(self, state_shape, num_actions, lr=1e-4, gamma=0.99, clip_param=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNetwork(state_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_param = clip_param
        self.entropy_coef = 0.01  # Added entropy coefficient for exploration
        
        self.memory = []
    
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, state_value = self.policy(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            dist = Categorical(logits=action_probs)
            action = dist.sample()
        
        return action.item(), state_value
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        if not self.memory:
            return
        
        states, actions, rewards, next_states, dones = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculate returns
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Calculate advantages
        _, values = self.policy(states)
        values = values.squeeze()
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
        action_probs, _ = self.policy(states)
        dist = Categorical(logits=action_probs)
        old_action_log_probs = dist.log_prob(actions).detach()
        
        # PPO loss
        for _ in range(4):  # Run multiple epochs of updates
            action_probs, values = self.policy(states)
            dist = Categorical(logits=action_probs)
            action_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            # Include entropy bonus for exploration
            loss = policy_loss + value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Gradient clipping
            self.optimizer.step()
        
        # Clear memory
        self.memory = []

class ImprovedTradingSignalGenerator:
    """Improved signal generator with better training and data"""
    
    def __init__(self, symbols=['EUR/USD', 'GBP/USD', 'USD/JPY']):
        self.symbols = symbols
        self.data_generator = RealisticDataGenerator()
        self.feature_engineer = EnhancedFeatureEngineer(window_size=50)
        
        # Initialize RL agent
        state_shape = (50, 19)  # window_size x num_features
        self.agent = ImprovedPPOAgent(state_shape, num_actions=3)
        
        # Load or train model
        self.model_path = "improved_trading_model.pth"
        if os.path.exists(self.model_path):
            try:
                self.load_model()
            except (RuntimeError, KeyError) as e:
                print(f"Model architecture mismatch: {e}")
                print("Training new model with updated architecture...")
                self.train_model()
        else:
            self.train_model()
    
    def fetch_data(self, symbol):
        """Fetch data using the realistic data generator"""
        return self.data_generator.generate_realistic_data(symbol, 1000, 'H1')
    
    def train_model(self):
        """Train the RL model with more episodes"""
        print("Training improved RL model...")
        
        # Use multiple symbols for training
        all_observations = []
        for symbol in self.symbols:
            sample_data = self.fetch_data(symbol)
            observations, _ = self.feature_engineer.prepare_features(sample_data)
            all_observations.extend(observations)
        
        # Use a subset for training if too large
        if len(all_observations) > 1000:
            all_observations = random.sample(all_observations, 1000)
            
        env = ImprovedAutoTradingEnv(all_observations)
        
        # Longer training with more exploration
        state = env.reset()
        total_reward = 0
        for step in range(5000):  # Increased training steps
            action, value = self.agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            self.agent.store_transition(state, action, reward, next_state, done)
            
            if done or step % 200 == 0:
                state = env.reset()
                if step % 1000 == 0:
                    print(f"Step {step}, Total Reward: {total_reward:.4f}")
                    total_reward = 0
            else:
                state = next_state
            
            if step % 100 == 0:
                self.agent.update()
        
        print("Training completed!")
        self.save_model()
    
    def save_model(self):
        """Save the trained model"""
        torch.save({
            'policy_state_dict': self.agent.policy.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'feature_dim': self.agent.policy.conv[0].weight.shape[1]
        }, self.model_path)
    
    def load_model(self):
        """Load a trained model with compatibility check"""
        checkpoint = torch.load(self.model_path)
        
        # Check if the saved model has the same feature dimension as the current model
        saved_feature_dim = checkpoint.get('feature_dim', None)
        current_feature_dim = self.agent.policy.conv[0].weight.shape[1]
        
        if saved_feature_dim is not None and saved_feature_dim != current_feature_dim:
            raise RuntimeError(f"Feature dimension mismatch: saved model has {saved_feature_dim}, current model expects {current_feature_dim}")
        
        self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def generate_signals(self):
        """Generate trading signals for all symbols"""
        all_signals = []
        
        for symbol in self.symbols:
            print(f"Generating signals for {symbol}...")
            
            # Fetch data
            df = self.fetch_data(symbol)
            
            # Prepare features
            observations, timestamps = self.feature_engineer.prepare_features(df)
            
            if len(observations) == 0:
                print(f"Not enough data for {symbol}")
                continue
            
            # Get the latest observation
            latest_obs = observations[-1]
            
            # Get prediction from RL model
            action, _ = self.agent.select_action(latest_obs, deterministic=True)
            
            # Current price
            current_price = df['Close'].iloc[-1]
            
            # Calculate stop loss and take profit based on ATR
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.01
            
            if action == 1:  # Buy signal
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (3 * atr)
                position_type = "BUY"
            elif action == 2:  # Sell signal
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (3 * atr)
                position_type = "SELL"
            else:  # Hold
                continue
            
            # Create signal
            signal = {
                'Time': timestamps[-1],
                'Currency': symbol,
                'Position type': position_type,
                'Buy price': current_price,
                'Take profit price': take_profit,
                'Stop loss price': stop_loss
            }
            
            all_signals.append(signal)
        
        return all_signals
    
    def signals_to_excel(self, signals, filename="trading_signals.xlsx"):
        """Save signals to Excel file"""
        if not signals:
            print("No signals to save")
            return
        
        df = pd.DataFrame(signals)
        
        # Format the DataFrame
        df['Time'] = pd.to_datetime(df['Time'])
        df['Buy price'] = df['Buy price'].round(5)
        df['Take profit price'] = df['Take profit price'].round(5)
        df['Stop loss price'] = df['Stop loss price'].round(5)
        
        # Save to Excel
        df.to_excel(filename, index=False)
        print(f"Signals saved to {filename}")
        
        return df

def main():
    """Main function with improved training and signal generation"""
    print("Starting Improved Reinforcement Learning Auto-Trading System...")
    
    # Initialize the signal generator
    signal_generator = ImprovedTradingSignalGenerator()
    
    # Run continuously and update every 10 minutes
    while True:
        try:
            print(f"\nGenerating signals at {datetime.now()}...")
            
            # Generate signals
            signals = signal_generator.generate_signals()
            
            # Save to Excel
            if signals:
                signal_generator.signals_to_excel(signals)
                print(f"Generated {len(signals)} signals")
                
                # Display the signals
                df = pd.DataFrame(signals)
                print("\nGenerated Signals:")
                print(df.to_string(index=False))
            else:
                print("No signals generated at this time.")
                # If no signals, try to retrain the model
                print("Retraining model to improve performance...")
                signal_generator.train_model()
            
            print("Waiting for next update in 10 minutes...")
            time.sleep(600)  # Wait for 10 minutes
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            print("Restarting in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    # Run the main function
    main()