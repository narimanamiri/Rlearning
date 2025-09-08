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
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class DataScraper:
    """Web scraper for financial data"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_yahoo_finance(self, symbol, period="1mo", interval="15m"):
        """Scrape data from Yahoo Finance using yfinance as fallback"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            return df
        except Exception as e:
            print(f"Error scraping Yahoo Finance: {e}")
            return self.generate_sample_data(symbol, 100)
    
    def scrape_investing_com(self, symbol):
        """Attempt to scrape from Investing.com (note: may be blocked)"""
        try:
            # This is a generic approach - would need to be customized for specific currency pairs
            url = f"https://www.investing.com/currencies/{symbol.replace('/', '-').lower()}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract price data - this would need to be customized based on the actual page structure
                price_element = soup.find('span', {'data-test': 'instrument-price-last'})
                if price_element:
                    price = float(price_element.text.replace(',', ''))
                    # For demonstration, we'll return sample data
                    return self.generate_sample_data(symbol, 100)
            return self.generate_sample_data(symbol, 100)
        except Exception as e:
            print(f"Error scraping Investing.com: {e}")
            return self.generate_sample_data(symbol, 100)
    
    def generate_sample_data(self, symbol, num_points):
        """Generate sample financial data when scraping fails"""
        dates = pd.date_range(end=datetime.now(), periods=num_points, freq='10T')
        base_price = 100 if 'USD' in symbol else 1.10
        
        prices = []
        current_price = base_price
        
        for _ in range(num_points):
            change = random.uniform(-0.001, 0.001) * current_price
            current_price += change
            prices.append(current_price)
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * random.uniform(1.0, 1.001) for p in prices],
            'Low': [p * random.uniform(0.999, 1.0) for p in prices],
            'Close': prices,
            'Volume': [random.randint(1000, 10000) for _ in range(num_points)]
        }, index=dates)
        
        return df

class FeatureEngineer:
    """Feature engineering for financial data"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        # Price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
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
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def normalize_features(self, df, feature_columns):
        """Normalize features using Z-score normalization"""
        for column in feature_columns:
            if column in df.columns:
                mean = df[column].mean()
                std = df[column].std()
                if std > 0:
                    df[column] = (df[column] - mean) / std
        return df
    
    def prepare_features(self, df):
        """Prepare all features for the RL model"""
        df = self.calculate_technical_indicators(df)
        
        feature_columns = [
            'returns', 'log_returns', 'sma_20', 'sma_50', 'macd', 
            'macd_signal', 'macd_hist', 'rsi', 'bb_width', 'atr', 'volume_ratio'
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

class AutoTradingEnv(gym.Env):
    """Custom Trading Environment for OpenAI Gym"""
    
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001, risk_free_rate=0.02/252):
        super(AutoTradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
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
        self.max_net_worth = self.initial_worth
        self.trades = 0
        self.current_step = 0
        self.done = False
        self.portfolio_history = []
        
        return self.data[self.current_step]
    
    def step(self, action):
        current_price = self.get_current_price()
        prev_net_worth = self.net_worth
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                # Buy with all available balance
                self.holdings = self.balance / current_price
                self.balance = 0
                self.trades += 1
        elif action == 2:  # Sell
            if self.holdings > 0:
                # Sell all holdings
                self.balance = self.holdings * current_price * (1 - self.transaction_cost)
                self.holdings = 0
                self.trades += 1
        
        # Update net worth
        self.net_worth = self.balance + self.holdings * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Calculate reward
        reward = self.calculate_reward(prev_net_worth)
        
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
    
    def get_current_price(self):
        # In a real implementation, this would get the actual price
        # For our simplified version, we'll use a proxy
        return 1.0  # This would be replaced with actual price data
    
    def calculate_reward(self, prev_net_worth):
        # Simple reward based on portfolio change
        portfolio_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        
        # Penalize excessive trading
        trade_penalty = 0.001 * self.trades
        
        # Risk-adjusted reward (simplified Sharpe ratio)
        risk_penalty = 0.1 * np.std(self.portfolio_history) if len(self.portfolio_history) > 1 else 0
        
        reward = portfolio_return - trade_penalty - risk_penalty
        return reward
    
    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}')

class PolicyNetwork(nn.Module):
    """Neural Network for Policy"""
    
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape[::-1]))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        # x shape: (batch_size, window_size, num_features)
        x = x.permute(0, 2, 1)  # Change to (batch_size, num_features, window_size)
        conv_out = self.conv(x).view(x.size(0), -1)
        action_probs = self.fc(conv_out)
        state_values = self.value_head(conv_out)
        return action_probs, state_values

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_shape, num_actions, lr=3e-4, gamma=0.99, clip_param=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNetwork(state_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_param = clip_param
        
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
            
            loss = policy_loss + value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.memory = []

class TradingSignalGenerator:
    """Main class to generate trading signals"""
    
    def __init__(self, symbols=['EUR/USD', 'GBP/USD', 'USD/JPY']):
        self.symbols = symbols
        self.scraper = DataScraper()
        self.feature_engineer = FeatureEngineer(window_size=50)
        
        # Initialize RL agent
        state_shape = (50, 11)  # window_size x num_features
        self.agent = PPOAgent(state_shape, num_actions=3)
        
        # Load or train model
        self.model_path = "trading_model.pth"
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.train_model()
    
    def fetch_data(self, symbol):
        """Fetch data for a symbol"""
        print(f"Fetching data for {symbol}...")
        
        # Try scraping from various sources
        df = self.scraper.scrape_yahoo_finance(symbol.replace('/', ''), period="1mo", interval="15m")
        
        if df is None or len(df) < 100:
            df = self.scraper.scrape_investing_com(symbol)
        
        return df
    
    def train_model(self):
        """Train the RL model"""
        print("Training RL model...")
        
        # For demonstration, we'll use sample data
        # In a real implementation, you would use historical data
        sample_data = self.scraper.generate_sample_data("EUR/USD", 1000)
        observations, _ = self.feature_engineer.prepare_features(sample_data)
        
        env = AutoTradingEnv(observations)
        
        # Simplified training loop
        state = env.reset()
        for step in range(1000):
            action, value = self.agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            self.agent.store_transition(state, action, reward, next_state, done)
            
            if done:
                state = env.reset()
            else:
                state = next_state
            
            if step % 50 == 0:
                self.agent.update()
        
        self.save_model()
    
    def save_model(self):
        """Save the trained model"""
        torch.save({
            'policy_state_dict': self.agent.policy.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
        }, self.model_path)
    
    def load_model(self):
        """Load a trained model"""
        checkpoint = torch.load(self.model_path)
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
    """Main function to run the trading signal generator"""
    print("Starting Reinforcement Learning Auto-Trading System...")
    
    # Initialize the signal generator
    signal_generator = TradingSignalGenerator()
    
    # Run continuously and update every 10 minutes
    while True:
        print(f"\nGenerating signals at {datetime.now()}...")
        
        # Generate signals
        signals = signal_generator.generate_signals()
        
        # Save to Excel
        signal_generator.signals_to_excel(signals)
        
        print("Waiting for next update in 10 minutes...")
        time.sleep(600)  # Wait for 10 minutes

if __name__ == "__main__":
    # For demonstration, we'll run once instead of continuously
    print("Starting Reinforcement Learning Auto-Trading System...")
    
    # Initialize the signal generator
    signal_generator = TradingSignalGenerator()
    
    # Generate signals once
    signals = signal_generator.generate_signals()
    
    # Save to Excel
    if signals:
        signal_generator.signals_to_excel(signals)
        print("Signals generated successfully!")
        
        # Display the signals
        df = pd.DataFrame(signals)
        print("\nGenerated Signals:")
        print(df.to_string(index=False))
    else:
        print("No signals generated at this time.")