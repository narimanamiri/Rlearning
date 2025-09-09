import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, config):
        self.window_size = config['env']['window_size']
        self.scalers = {}

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_30'] = df['close'].rolling(30).mean()
        df['rsi'] = self._compute_rsi(df['close'], 14)
        macd, signal = self._compute_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        bb_upper, bb_lower = self._compute_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['atr'] = self._compute_atr(df, 14)
        df['volatility'] = df['log_return'].rolling(10).std()
        return df

    def _compute_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def _compute_bollinger_bands(self, prices, window=20, num_std=2):
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower

    def _compute_atr(self, df, window=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        return atr

    def normalize_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        df = df.copy()
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        if fit_scaler:
            self.scalers = {}
            for col in feature_cols:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]].fillna(0))
                self.scalers[col] = scaler
        else:
            for col in feature_cols:
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[[col]].fillna(0))
        return df

    def create_windows(self, df: pd.DataFrame) -> np.ndarray:
        df = df.dropna()
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        data = df[feature_cols].values
        windows = []
        for i in range(self.window_size, len(data)):
            window = data[i - self.window_size:i]
            windows.append(window)
        return np.array(windows)

    def get_feature_dim(self, df: pd.DataFrame) -> int:
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        return len(feature_cols)