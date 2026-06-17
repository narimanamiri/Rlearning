"""
Comprehensive Crypto Trading System with Reinforcement Learning & MetaTrader Integration
Version: 3.0
Author: AI Trading Engineer
Date: 2025-09-09
Description: End-to-end trading system with MT5 integration, RL, and adaptive learning
"""

# Core imports
import os
import time
import random
import json
import logging
import warnings
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Data processing imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
from scipy.fft import fft, ifft
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# MetaTrader 5 integration
import MetaTrader5 as mt5

# Image processing imports
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import easyocr

# Web scraping imports
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import cloudscraper

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
from torchvision import models, transforms

# Reinforcement learning imports
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# API imports
import ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoTrader")

# Suppress warnings
warnings.filterwarnings('ignore')


# Custom exceptions
class TradingSystemError(Exception):
    """Base exception for the trading system."""


class DataAcquisitionError(TradingSystemError):
    """Raised when no data source can supply data for a symbol."""


class ModelTrainingError(TradingSystemError):
    """Raised when RL model training fails."""

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
class Config:
    """System configuration settings"""
    
    # MetaTrader 5 settings
    MT5_LOGIN = 0  # Your MT5 account number
    MT5_PASSWORD = ""  # Your MT5 password
    MT5_SERVER = ""  # Your broker's server
    MT5_TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute timeframe
    MT5_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]  # Symbols to trade
    
    # Trading parameters
    INITIAL_BALANCE = 10000.0
    TRANSACTION_COST = 0.001  # 0.1% transaction cost
    RISK_FREE_RATE = 0.02 / 252  # Daily risk-free rate
    
    # RL training parameters
    TRAINING_STEPS = 100000
    N_ENVS = 4  # Number of parallel environments
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    N_EPOCHS = 10
    
    # Technical indicator parameters
    SMA_WINDOWS = [5, 10, 20, 50]
    EMA_WINDOWS = [12, 26]
    RSI_WINDOW = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_WINDOW = 20
    BB_STD = 2
    ATR_WINDOW = 14
    
    # Feature engineering
    WINDOW_SIZE = 50
    N_FEATURES = 30
    FEATURE_SCALER = "standard"  # "standard" or "robust"
    
    # Adaptive learning parameters
    LEARNING_MEMORY_SIZE = 10000
    REWARD_THRESHOLD = 0.7  # Minimum reward threshold for model updates
    LOSS_THRESHOLD = 0.3  # Maximum loss threshold for strategy adjustment
    
    # File paths
    DATA_DIR = Path("./data")
    MODELS_DIR = Path("./models")
    LOGS_DIR = Path("./logs")
    CHARTS_DIR = Path("./charts")
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, CHARTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Enhanced DataAcquirer with MetaTrader 5 integration
class DataAcquirer:
    """
    Comprehensive data acquisition system with MetaTrader 5 integration
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.mt5_connected = False
        self.init_mt5()
        
        # Initialize other data sources as fallback
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        logger.info("DataAcquirer initialized with MetaTrader 5 integration")
    
    def init_mt5(self):
        """Initialize MetaTrader 5 connection"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login to MT5 account if credentials provided
            if self.config.MT5_PASSWORD and self.config.MT5_SERVER:
                authorized = mt5.login(
                    self.config.MT5_LOGIN,
                    self.config.MT5_PASSWORD,
                    self.config.MT5_SERVER
                )
                if authorized:
                    logger.info("MT5 login successful")
                else:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
            
            self.mt5_connected = True
            logger.info("MetaTrader 5 initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            self.mt5_connected = False
            return False
    
    def get_historical_data(self, symbol: str, timeframe: str = '1m', 
                          limit: int = 1000) -> pd.DataFrame:
        """
        Get historical OHLCV data from MetaTrader 5 with fallback to other sources
        """
        # Try MT5 first
        if self.mt5_connected and timeframe == '1m':
            df = self._get_from_mt5(symbol, limit)
            if df is not None and not df.empty:
                return df
        
        # Fallback to other methods
        methods = [
            self._get_from_binance_api,
            self._get_from_ccxt,
            self._generate_fallback_data
        ]
        
        for method in methods:
            try:
                df = method(symbol, timeframe, limit)
                if df is not None and not df.empty:
                    logger.info(f"Successfully acquired data for {symbol} using {method.__name__}")
                    return df
            except Exception as e:
                logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        raise DataAcquisitionError(f"All data acquisition methods failed for {symbol}")
    
    def _get_from_mt5(self, symbol: str, limit: int) -> pd.DataFrame:
        """Get 1-minute data from MetaTrader 5"""
        try:
            # Convert symbol to MT5 format if needed
            mt5_symbol = self._convert_symbol_to_mt5(symbol)
            
            # Get current rates
            rates = mt5.copy_rates_from_pos(mt5_symbol, self.config.MT5_TIMEFRAME, 0, limit)
            
            if rates is None:
                logger.warning(f"MT5 returned no data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Rename columns to match expected format
            df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            # Select only necessary columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Retrieved {len(df)} 1-minute bars from MT5 for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"MT5 data acquisition failed: {e}")
            return None
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time tick data from MT5"""
        try:
            if not self.mt5_connected:
                return None
            
            mt5_symbol = self._convert_symbol_to_mt5(symbol)
            tick = mt5.symbol_info_tick(mt5_symbol)
            
            if tick is None:
                return None
            
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': pd.to_datetime(tick.time, unit='s'),
                'spread': tick.ask - tick.bid
            }
            
        except Exception as e:
            logger.error(f"Real-time data acquisition failed: {e}")
            return None
    
    def _convert_symbol_to_mt5(self, symbol: str) -> str:
        """Convert standard symbol format to MT5 format"""
        symbol_map = {
            'BTC/USD': 'BTCUSD',
            'ETH/USD': 'ETHUSD',
            'EUR/USD': 'EURUSD',
            'GBP/USD': 'GBPUSD',
            'USD/JPY': 'USDJPY',
            'XAU/USD': 'XAUUSD'
        }
        return symbol_map.get(symbol, symbol.replace('/', ''))
    
    # Keep other existing methods (_get_from_binance_api, _get_from_ccxt, etc.)
    def _get_from_binance_api(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get data from Binance API (fallback method)"""
        try:
            # Implementation similar to previous version
            binance_symbol = symbol.replace('/', '').upper()
            if 'USD' in binance_symbol and not binance_symbol.endswith('USDT'):
                binance_symbol += 'USDT'
            
            # Use python-binance or ccxt for fallback
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(binance_symbol, timeframe, limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"Binance API failed: {e}")
            return None
    
    def _get_from_ccxt(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get data using CCXT library"""
        try:
            exchange = ccxt.binance()
            exchange_symbol = symbol.replace('/', '')
            
            ohlcv = exchange.fetch_ohlcv(exchange_symbol, timeframe, limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"CCXT failed: {e}")
            return None
    
    def _generate_fallback_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Generate synthetic data when all other sources fail"""
        try:
            logger.warning(f"Generating fallback data for {symbol}")
            
            # Create realistic synthetic data with 1-minute intervals
            base_price = 10000 if 'BTC' in symbol else 1.0
            volatility = 0.001  # 0.1% volatility per minute
            
            dates = pd.date_range(end=datetime.now(), periods=limit, freq='1min')
            prices = []
            current_price = base_price
            
            for _ in range(limit):
                # Random walk with mean reversion
                change = np.random.normal(0, volatility) * current_price
                current_price += change
                
                # Add slight mean reversion
                current_price = current_price * 0.9999 + base_price * 0.0001
                
                prices.append(current_price)
            
            # Create OHLC data from prices
            opens = [p * (1 + np.random.uniform(-0.0005, 0.0005)) for p in prices]
            highs = [max(o, p) * (1 + np.random.uniform(0, 0.001)) for o, p in zip(opens, prices)]
            lows = [min(o, p) * (1 - np.random.uniform(0, 0.001)) for o, p in zip(opens, prices)]
            closes = prices
            volumes = [np.random.lognormal(8, 1) for _ in prices]
            
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=dates)
            
            return df
            
        except Exception as e:
            logger.error(f"Fallback data generation failed: {e}")
            return None
    
    def __del__(self):
        """Cleanup method"""
        if self.mt5_connected:
            mt5.shutdown()

# Enhanced Trading Environment with Adaptive Learning
class CryptoTradingEnvironment(gym.Env):
    """
    Enhanced trading environment with MetaTrader 5 integration and adaptive learning
    """
    
    def __init__(self, symbol: str, data_acquirer: DataAcquirer, config: Config, 
                 initial_balance: float = 10000.0, lookback_window: int = 50):
        super().__init__()
        
        self.symbol = symbol
        self.data_acquirer = data_acquirer
        self.config = config
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Initialize state
        self.current_step = 0
        self.balance = initial_balance
        self.holdings = 0.0
        self.total_value = initial_balance
        self.previous_value = initial_balance
        self.done = False
        
        # Trading history
        self.trades = []
        self.portfolio_history = []
        self.reward_history = []
        
        # Load initial data
        self.data = self._load_data()
        self.n_steps = len(self.data) - lookback_window - 1
        
        # Define action and observation space
        # Action: 0=Hold, 1=Buy, 2=Sell, 3=Close Position
        self.action_space = spaces.Discrete(4)
        
        # Observation: OHLCV + technical indicators + portfolio state.
        # _get_features returns a flat 1-D vector, so the space must be 1-D to
        # match (a 2-D shape here makes SB3 reject every observation).
        n_features = len(self._get_features(self.lookback_window))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Adaptive learning parameters
        self.learning_memory = []
        self.win_rate = 0.5
        self.reward_threshold = config.REWARD_THRESHOLD
        self.loss_threshold = config.LOSS_THRESHOLD
        
        logger.info(f"Trading environment initialized for {symbol}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and prepare trading data"""
        try:
            # Get 1-minute data
            df = self.data_acquirer.get_historical_data(
                self.symbol, '1m', 5000  # Get more data for buffer
            )
            
            if df is None or df.empty:
                raise DataAcquisitionError(f"No data available for {self.symbol}")
            
            # Calculate technical indicators
            feature_engineer = FeatureEngineer(self.config)
            df = feature_engineer.calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Generate fallback data
            return self.data_acquirer._generate_fallback_data(self.symbol, '1m', 5000)
    
    def _get_features(self, step: int) -> np.ndarray:
        """Extract features for current step"""
        if step < self.lookback_window:
            step = self.lookback_window
        
        current_data = self.data.iloc[step - self.lookback_window:step]

        # Use the actual number of rows for the zero-fallbacks so that every
        # sub-array has the same length; otherwise np.concatenate below raises
        # when an indicator column is missing but `close`/`volume` are present.
        n_rows = len(current_data)

        # Price features
        features = []
        features.extend([
            current_data['close'].values,
            current_data['volume'].values,
            current_data['returns'].values if 'returns' in current_data.columns else np.zeros(n_rows),
            current_data['rsi'].values if 'rsi' in current_data.columns else np.zeros(n_rows),
            current_data['macd'].values if 'macd' in current_data.columns else np.zeros(n_rows),
        ])
        
        # Flatten and add portfolio state
        flat_features = np.concatenate([f.flatten() for f in features])
        
        # Add portfolio state
        portfolio_features = np.array([
            self.balance / self.initial_balance,
            self.holdings,
            self.total_value / self.initial_balance,
            len(self.trades) / 100.0  # Normalized trade count
        ])
        
        return np.concatenate([flat_features, portfolio_features])
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.total_value = self.initial_balance
        self.previous_value = self.initial_balance
        self.done = False
        
        self.trades = []
        self.portfolio_history = []
        self.reward_history = []
        
        # Reload data for new episode
        self.data = self._load_data()
        self.n_steps = len(self.data) - self.lookback_window - 1
        
        observation = self._get_features(self.current_step)
        info = {}
        
        return observation.astype(np.float32), info
    
    def step(self, action: int):
        """Execute trading action"""
        if self.done:
            raise ValueError("Episode has ended. Call reset() first.")
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = 0.0
        transaction_cost = self.config.TRANSACTION_COST
        
        if action == 1:  # Buy
            if self.balance > current_price:
                # Calculate maximum affordable units
                max_units = self.balance / (current_price * (1 + transaction_cost))
                units_to_buy = max_units * 0.1  # Risk management: only use 10% of balance
                
                cost = units_to_buy * current_price * (1 + transaction_cost)
                if cost <= self.balance:
                    self.holdings += units_to_buy
                    self.balance -= cost
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'units': units_to_buy,
                        'price': current_price,
                        'cost': cost
                    })
        
        elif action == 2:  # Sell
            if self.holdings > 0:
                # Sell all holdings
                revenue = self.holdings * current_price * (1 - transaction_cost)
                self.balance += revenue
                self.trades.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'units': self.holdings,
                    'price': current_price,
                    'revenue': revenue
                })
                self.holdings = 0.0
        
        elif action == 3:  # Close Position
            if self.holdings > 0:
                revenue = self.holdings * current_price * (1 - transaction_cost)
                self.balance += revenue
                self.holdings = 0.0
        
        # Update portfolio value
        self.previous_value = self.total_value
        self.total_value = self.balance + (self.holdings * current_price)
        
        # Calculate reward
        reward = self._calculate_reward(action, current_price)
        self.reward_history.append(reward)
        
        # Update adaptive learning parameters
        self._update_learning_parameters(reward)
        
        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= self.n_steps or self.total_value <= self.initial_balance * 0.5  # Stop if loss > 50%
        
        # Get next observation
        observation = self._get_features(self.current_step)
        
        # Record portfolio state
        self.portfolio_history.append({
            'step': self.current_step,
            'balance': self.balance,
            'holdings': self.holdings,
            'total_value': self.total_value,
            'price': current_price
        })
        
        info = {
            'current_price': current_price,
            'balance': self.balance,
            'holdings': self.holdings,
            'total_value': self.total_value,
            'reward': reward,
            'win_rate': self.win_rate
        }
        
        return observation.astype(np.float32), reward, self.done, False, info
    
    def _calculate_reward(self, action: int, current_price: float) -> float:
        """Calculate reward with adaptive learning"""
        # Basic reward based on portfolio value change
        value_change = (self.total_value - self.previous_value) / self.previous_value
        
        # Action-specific rewards/penalties
        action_reward = 0.0
        
        if action == 0:  # Hold
            # Small penalty for inactivity during trends
            if len(self.data) > self.current_step + 1:
                future_price = self.data.iloc[self.current_step + 1]['close']
                price_change = (future_price - current_price) / current_price
                if abs(price_change) > 0.001:  # If significant movement
                    action_reward = -0.01 * np.sign(price_change)  # Penalize missing movement
        
        elif action in [1, 2]:  # Buy or Sell
            # Reward/punish based on immediate outcome
            if len(self.data) > self.current_step + 5:  # Look 5 steps ahead
                future_price = self.data.iloc[self.current_step + 5]['close']
                price_change = (future_price - current_price) / current_price
                
                if action == 1:  # Buy
                    action_reward = price_change * 10  # Amplify reward for correct buy
                else:  # Sell
                    action_reward = -price_change * 10  # Reward for correct sell
        
        # Risk-adjusted reward
        risk_penalty = -0.01 * (len(self.trades) / 100.0)  # Penalize overtrading
        
        # Combine rewards
        total_reward = value_change * 100 + action_reward + risk_penalty
        
        # Adaptive reward scaling based on performance
        if len(self.reward_history) > 10:
            recent_performance = np.mean(self.reward_history[-10:])
            if recent_performance < self.loss_threshold:
                total_reward *= 1.5  # Increase sensitivity during poor performance
            elif recent_performance > self.reward_threshold:
                total_reward *= 0.8  # Reduce sensitivity during good performance
        
        return float(total_reward)
    
    def _update_learning_parameters(self, reward: float):
        """Update adaptive learning parameters based on performance"""
        # Store reward in memory
        self.learning_memory.append({
            'step': self.current_step,
            'reward': reward,
            'timestamp': datetime.now()
        })
        
        # Keep memory size limited
        if len(self.learning_memory) > self.config.LEARNING_MEMORY_SIZE:
            self.learning_memory = self.learning_memory[-self.config.LEARNING_MEMORY_SIZE:]
        
        # Update win rate (simplified)
        if len(self.reward_history) > 0:
            positive_rewards = sum(1 for r in self.reward_history if r > 0)
            self.win_rate = positive_rewards / len(self.reward_history)
        
        # Adjust reward threshold based on performance
        if len(self.reward_history) > 100:
            recent_avg_reward = np.mean(self.reward_history[-100:])
            if recent_avg_reward > self.reward_threshold:
                self.reward_threshold = min(0.9, self.reward_threshold * 1.01)  # Gradually increase
            else:
                self.reward_threshold = max(0.3, self.reward_threshold * 0.99)  # Gradually decrease

# Enhanced RL Agent with Adaptive Learning
class AdaptiveTradingAgent:
    """
    Reinforcement learning agent that adapts based on rewards and losses
    """
    
    def __init__(self, config: Config, symbol: str):
        self.config = config
        self.symbol = symbol
        self.data_acquirer = DataAcquirer(config)
        
        # Initialize environment
        self.env = CryptoTradingEnvironment(symbol, self.data_acquirer, config)
        
        # RL model
        self.model = None
        self.is_trained = False
        
        # Learning statistics
        self.learning_history = []
        self.performance_metrics = {
            'total_rewards': [],
            'win_rates': [],
            'portfolio_values': [],
            'learning_rates': []
        }
        
        logger.info(f"AdaptiveTradingAgent initialized for {symbol}")
    
    def create_model(self):
        """Create RL model with adaptive architecture"""
        policy_kwargs = dict(
            net_arch=[256, 256, 128, 64]  # Neural network architecture
        )
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config.LEARNING_RATE,
            n_steps=2048,
            batch_size=self.config.BATCH_SIZE,
            n_epochs=self.config.N_EPOCHS,
            gamma=self.config.GAMMA,
            gae_lambda=self.config.GAE_LAMBDA,
            ent_coef=self.config.ENT_COEF,
            vf_coef=self.config.VF_COEF,
            max_grad_norm=self.config.MAX_GRAD_NORM,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(self.config.LOGS_DIR / "tensorboard")
        )
        
        logger.info("RL model created successfully")
    
    def train(self, total_timesteps: int = None):
        """Train the agent with adaptive learning"""
        if total_timesteps is None:
            total_timesteps = self.config.TRAINING_STEPS
        
        if self.model is None:
            self.create_model()
        
        # Custom callback for adaptive learning
        class AdaptiveCallback(BaseCallback):
            def __init__(self, agent, verbose=0):
                super().__init__(verbose)
                self.agent = agent
            
            def _on_step(self) -> bool:
                # Adaptive learning rate adjustment
                if len(self.agent.performance_metrics['total_rewards']) > 10:
                    recent_rewards = self.agent.performance_metrics['total_rewards'][-10:]
                    avg_reward = np.mean(recent_rewards)
                    
                    # Adjust learning rate based on performance
                    if avg_reward < self.agent.config.LOSS_THRESHOLD:
                        new_lr = self.model.learning_rate * 1.1  # Increase LR if performing poorly
                        self.model.learning_rate = min(new_lr, 1e-2)
                    elif avg_reward > self.agent.config.REWARD_THRESHOLD:
                        new_lr = self.model.learning_rate * 0.9  # Decrease LR if performing well
                        self.model.learning_rate = max(new_lr, 1e-5)
                
                return True
        
        # Train the model
        callback = AdaptiveCallback(self)
        
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                reset_num_timesteps=False
            )
            
            self.is_trained = True
            logger.info("Training completed successfully")
            
            # Save the trained model
            self.save_model()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise ModelTrainingError(f"Training failed: {e}")
    
    def predict_action(self, observation: np.ndarray) -> int:
        """Predict trading action with exploration"""
        if self.model is None or not self.is_trained:
            return random.randint(0, 3)  # Random action if not trained
        
        # Add exploration based on recent performance
        exploration_rate = self._get_exploration_rate()
        
        if random.random() < exploration_rate:
            return random.randint(0, 3)  # Explore
        
        # Exploit: use trained model
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
    
    def _get_exploration_rate(self) -> float:
        """Calculate adaptive exploration rate"""
        base_rate = 0.1  # 10% base exploration
        
        if len(self.performance_metrics['win_rates']) > 0:
            recent_win_rate = np.mean(self.performance_metrics['win_rates'][-10:])
            
            # Increase exploration if performance is poor
            if recent_win_rate < 0.4:
                return min(0.5, base_rate * 2.0)
            # Decrease exploration if performance is good
            elif recent_win_rate > 0.7:
                return max(0.01, base_rate * 0.5)
        
        return base_rate
    
    def update_with_feedback(self, reward: float, action: int, outcome: float):
        """Update agent based on trading feedback"""
        feedback = {
            'timestamp': datetime.now(),
            'reward': reward,
            'action': action,
            'outcome': outcome,
            'symbol': self.symbol
        }
        
        self.learning_history.append(feedback)
        
        # If feedback is strongly negative, trigger retraining
        if reward < -0.5 and len(self.learning_history) > 100:
            logger.info("Negative feedback detected, triggering adaptive retraining...")
            self.adaptive_retrain()
    
    def adaptive_retrain(self, additional_timesteps: int = 10000):
        """Retrain model adaptively based on recent performance"""
        if not self.is_trained:
            logger.warning("Model not trained yet, cannot perform adaptive retraining")
            return
        
        logger.info("Starting adaptive retraining...")
        
        # Adjust learning parameters based on recent performance
        recent_rewards = [fh['reward'] for fh in self.learning_history[-100:]]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        if avg_reward < self.config.LOSS_THRESHOLD:
            # Poor performance: increase learning rate and retrain
            self.model.learning_rate = min(self.model.learning_rate * 1.5, 1e-2)
            logger.info(f"Increased learning rate to {self.model.learning_rate}")
        
        # Retrain with additional timesteps
        self.train(additional_timesteps)
    
    def save_model(self):
        """Save trained model"""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        model_path = self.config.MODELS_DIR / f"trading_agent_{self.symbol.replace('/', '_')}.zip"
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load trained model"""
        model_path = self.config.MODELS_DIR / f"trading_agent_{self.symbol.replace('/', '_')}.zip"
        
        if not model_path.exists():
            logger.warning(f"No saved model found at {model_path}")
            return False
        
        if self.model is None:
            self.create_model()
        
        self.model = PPO.load(model_path, env=self.env)
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")
        return True
    
    def evaluate_performance(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        total_rewards = []
        final_values = []
        win_rates = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_rewards = []
            done = False
            
            while not done:
                action = self.predict_action(obs)
                obs, reward, done, _, info = self.env.step(action)
                episode_rewards.append(reward)
            
            total_rewards.append(sum(episode_rewards))
            final_values.append(info['total_value'])
            
            # Calculate win rate for this episode
            positive_rewards = sum(1 for r in episode_rewards if r > 0)
            win_rates.append(positive_rewards / len(episode_rewards) if episode_rewards else 0)
        
        metrics = {
            'avg_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            'avg_final_value': np.mean(final_values),
            'avg_win_rate': np.mean(win_rates),
            'sharpe_ratio': np.mean(total_rewards) / (np.std(total_rewards) + 1e-8),
            'max_drawdown': min(total_rewards) if total_rewards else 0
        }
        
        # Update performance metrics
        self.performance_metrics['total_rewards'].extend(total_rewards)
        self.performance_metrics['win_rates'].extend(win_rates)
        self.performance_metrics['portfolio_values'].extend(final_values)
        
        logger.info(f"Performance evaluation completed: {metrics}")
        return metrics

# Main Trading System
class CryptoTradingSystem:
    """
    Main trading system that coordinates data acquisition, RL agents, and trading execution
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_acquirer = DataAcquirer(config)
        self.agents = {}
        self.is_running = False
        
        logger.info("CryptoTradingSystem initialized")
    
    def initialize_agents(self, symbols: List[str] = None):
        """Initialize trading agents for specified symbols"""
        if symbols is None:
            symbols = self.config.MT5_SYMBOLS
        
        for symbol in symbols:
            try:
                self.agents[symbol] = AdaptiveTradingAgent(self.config, symbol)
                logger.info(f"Initialized trading agent for {symbol}")
            except Exception as e:
                logger.error(f"Failed to initialize agent for {symbol}: {e}")
    
    def train_agents(self, training_steps: int = None):
        """Train all initialized agents"""
        if training_steps is None:
            training_steps = self.config.TRAINING_STEPS
        
        for symbol, agent in self.agents.items():
            try:
                logger.info(f"Training agent for {symbol}...")
                agent.train(training_steps)
                
                # Evaluate initial performance
                metrics = agent.evaluate_performance()
                logger.info(f"Initial performance for {symbol}: {metrics}")
                
            except Exception as e:
                logger.error(f"Training failed for {symbol}: {e}")
    
    def start_trading(self, live_mode: bool = False):
        """Start the trading system"""
        self.is_running = True
        
        logger.info("Starting trading system..." + (" LIVE MODE" if live_mode else " SIMULATION MODE"))
        
        try:
            while self.is_running:
                for symbol, agent in self.agents.items():
                    if not agent.is_trained:
                        logger.warning(f"Agent for {symbol} not trained, skipping")
                        continue
                    
                    # Get real-time data
                    if live_mode:
                        realtime_data = self.data_acquirer.get_realtime_data(symbol)
                        if realtime_data:
                            # Convert to observation format (simplified)
                            observation = self._create_observation_from_realtime(realtime_data, symbol)
                            
                            # Get action from agent
                            action = agent.predict_action(observation)
                            
                            # Execute trade (in simulation or live)
                            self._execute_trade(symbol, action, realtime_data, live_mode)
                    
                    else:
                        # Simulation mode - use historical data
                        self._run_simulation_step(symbol, agent)
                
                # Adaptive learning check
                self._perform_adaptive_learning()
                
                # Brief pause to avoid overwhelming the system
                time.sleep(1 if live_mode else 0.1)
                
        except KeyboardInterrupt:
            logger.info("Trading system stopped by user")
        except Exception as e:
            logger.error(f"Trading system error: {e}")
        finally:
            self.is_running = False
    
    def _create_observation_from_realtime(self, realtime_data: Dict, symbol: str) -> np.ndarray:
        """Create observation from real-time data"""
        # This is a simplified implementation
        # In practice, you'd want to maintain a rolling window of recent data
        observation = np.array([
            realtime_data['bid'],
            realtime_data['ask'],
            realtime_data['spread'],
            realtime_data['volume'] if realtime_data['volume'] else 0
        ])
        
        return observation.astype(np.float32)
    
    def _execute_trade(self, symbol: str, action: int, data: Dict, live_mode: bool):
        """Execute a trade (simulated or live)"""
        try:
            if live_mode:
                # Live trading execution via MT5
                if action == 1:  # Buy
                    result = self._execute_mt5_order(symbol, mt5.ORDER_TYPE_BUY, data['ask'])
                elif action == 2:  # Sell
                    result = self._execute_mt5_order(symbol, mt5.ORDER_TYPE_SELL, data['bid'])
                elif action == 3:  # Close
                    result = self._close_mt5_position(symbol)
                else:  # Hold
                    return
                
                logger.info(f"Live trade executed: {symbol} Action={action} Result={result}")
            else:
                # Simulated trading
                logger.info(f"Simulated trade: {symbol} Action={action} Price={data.get('bid', data.get('last', 0))}")
                
        except Exception as e:
            logger.error(f"Trade execution failed for {symbol}: {e}")
    
    def _execute_mt5_order(self, symbol: str, order_type: int, price: float, volume: float = 0.1):
        """Execute order via MetaTrader 5"""
        if not mt5.initialize():
            logger.error("MT5 not initialized")
            return None
        
        mt5_symbol = self.data_acquirer._convert_symbol_to_mt5(symbol)
        
        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "RL Trading System",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(order_request)
        return result
    
    def _close_mt5_position(self, symbol: str):
        """Close position via MetaTrader 5"""
        mt5_symbol = self.data_acquirer._convert_symbol_to_mt5(symbol)
        
        positions = mt5.positions_get(symbol=mt5_symbol)
        if not positions:
            return None
        
        for position in positions:
            order_type = mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(mt5_symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(mt5_symbol).ask
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": 10,
                "magic": 234000,
                "comment": "Close RL Position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            return result
        
        return None
    
    def _run_simulation_step(self, symbol: str, agent: AdaptiveTradingAgent):
        """Run one simulation step"""
        try:
            # This would integrate with the environment's step function
            # Simplified implementation for demonstration
            pass
        except Exception as e:
            logger.error(f"Simulation step failed for {symbol}: {e}")
    
    def _perform_adaptive_learning(self):
        """Perform adaptive learning based on recent performance"""
        for symbol, agent in self.agents.items():
            if len(agent.learning_history) > 100:
                recent_rewards = [fh['reward'] for fh in agent.learning_history[-100:]]
                avg_reward = np.mean(recent_rewards)
                
                if avg_reward < self.config.LOSS_THRESHOLD:
                    logger.info(f"Poor performance detected for {symbol}, triggering adaptive learning")
                    agent.adaptive_retrain(5000)  # Additional training
    
    def stop_trading(self):
        """Stop the trading system"""
        self.is_running = False
        logger.info("Trading system stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for all agents"""
        report = {
            'timestamp': datetime.now(),
            'agents': {},
            'overall_metrics': {}
        }
        
        total_rewards = []
        win_rates = []
        portfolio_values = []
        
        for symbol, agent in self.agents.items():
            agent_report = agent.evaluate_performance()
            report['agents'][symbol] = agent_report
            
            total_rewards.extend(agent.performance_metrics['total_rewards'][-10:])
            win_rates.extend(agent.performance_metrics['win_rates'][-10:])
            portfolio_values.extend(agent.performance_metrics['portfolio_values'][-10:])
        
        if total_rewards:
            report['overall_metrics'] = {
                'avg_total_reward': np.mean(total_rewards),
                'avg_win_rate': np.mean(win_rates),
                'avg_portfolio_value': np.mean(portfolio_values),
                'system_health': 'GOOD' if np.mean(win_rates) > 0.5 else 'NEEDS_ATTENTION'
            }
        
        return report

# FeatureEngineer class (keep from original implementation)
class FeatureEngineer:
    """Feature engineering class (implementation similar to original)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler() if config.FEATURE_SCALER == "standard" else RobustScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.pca = PCA(n_components=0.95)
        logger.info("FeatureEngineer initialized")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in self.config.SMA_WINDOWS:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        for window in self.config.EMA_WINDOWS:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.config.RSI_WINDOW)
        
        # MACD
        macd, signal, hist = self._calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=self.config.MACD_FAST, adjust=False).mean()
        ema_slow = prices.ewm(span=self.config.MACD_SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.config.MACD_SIGNAL, adjust=False).mean()
        return macd, signal, macd - signal

# Usage example
if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Initialize trading system
    trading_system = CryptoTradingSystem(config)
    
    # Initialize agents for specified symbols
    trading_system.initialize_agents(["EURUSD", "BTCUSD"])
    
    # Train agents
    print("Training agents...")
    trading_system.train_agents(training_steps=50000)  # Reduced for demo
    
    # Start trading in simulation mode
    print("Starting simulation trading...")
    
    # Run for a short period in simulation
    import threading
    
    def run_simulation():
        trading_system.start_trading(live_mode=False)
    
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.start()
    
    # Let it run for 30 seconds
    time.sleep(30)
    
    # Stop the system
    trading_system.stop_trading()
    simulation_thread.join()
    
    # Generate performance report
    report = trading_system.get_performance_report()
    print("\n=== Performance Report ===")
    print(json.dumps(report, indent=2, default=str))
    
    # Shutdown MT5
    if mt5.initialize():
        mt5.shutdown()