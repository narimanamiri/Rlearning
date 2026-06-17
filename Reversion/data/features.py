import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Names accepted in config ``data.indicators`` / the ``--indicators`` CLI flag.
# Each maps to a small group of columns added by ``add_technical_indicators``.
# "base" reproduces the original indicator set; the rest are opt-in extras.
AVAILABLE_INDICATORS = (
    "base",        # log_return, sma_10, sma_30, rsi, macd(+signal),
                   # bollinger bands, atr, volatility
    "stochastic",  # stochastic oscillator %K / %D
    "obv",         # on-balance volume
    "roc",         # rate of change (momentum)
    "williams_r",  # Williams %R
    "cci",         # commodity channel index
)

DEFAULT_INDICATORS = ("base",)


class FeatureEngineer:
    def __init__(self, config):
        self.window_size = config['env']['window_size']
        self.scalers = {}
        # Selected indicator groups. Falls back to the original "base" set so
        # existing configs keep working unchanged.
        requested = (config.get('data', {}) or {}).get('indicators') or DEFAULT_INDICATORS
        if isinstance(requested, str):
            requested = [requested]
        unknown = [i for i in requested if i not in AVAILABLE_INDICATORS]
        if unknown:
            raise ValueError(
                f"Unknown indicator(s): {unknown}. "
                f"Available: {list(AVAILABLE_INDICATORS)}")
        # Always keep "base" so there is at least a price/return signal.
        self.indicators = list(dict.fromkeys(["base", *requested]))

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "base" in self.indicators:
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
        if "stochastic" in self.indicators:
            k, d = self._compute_stochastic(df)
            df['stoch_k'] = k
            df['stoch_d'] = d
        if "obv" in self.indicators:
            df['obv'] = self._compute_obv(df)
        if "roc" in self.indicators:
            df['roc'] = self._compute_roc(df['close'], 10)
        if "williams_r" in self.indicators:
            df['williams_r'] = self._compute_williams_r(df, 14)
        if "cci" in self.indicators:
            df['cci'] = self._compute_cci(df, 20)
        return df

    def _compute_stochastic(self, df, k_window=14, d_window=3):
        low_min = df['low'].rolling(k_window).min()
        high_max = df['high'].rolling(k_window).max()
        rng = (high_max - low_min).replace(0, np.nan)
        k = 100 * (df['close'] - low_min) / rng
        d = k.rolling(d_window).mean()
        return k, d

    def _compute_obv(self, df):
        direction = np.sign(df['close'].diff().fillna(0.0))
        return (direction * df['volume']).cumsum()

    def _compute_roc(self, prices, window=10):
        return prices.pct_change(periods=window) * 100.0

    def _compute_williams_r(self, df, window=14):
        high_max = df['high'].rolling(window).max()
        low_min = df['low'].rolling(window).min()
        rng = (high_max - low_min).replace(0, np.nan)
        return -100 * (high_max - df['close']) / rng

    def _compute_cci(self, df, window=20):
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        sma = tp.rolling(window).mean()
        mean_dev = tp.rolling(window).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        mean_dev = mean_dev.replace(0, np.nan)
        return (tp - sma) / (0.015 * mean_dev)

    def _compute_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        # Avoid divide-by-zero when there are no losses in the window: RSI -> 100.
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(loss != 0, 100.0)
        return rsi

    def _compute_macd(self, prices, fast=12, slow=26, signal=9):
        # adjust=False gives the standard recursive EMA used for MACD.
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
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

    @staticmethod
    def _feature_cols(df: pd.DataFrame):
        return [c for c in df.columns
                if c not in ['open', 'high', 'low', 'close', 'volume']]

    def fit_scalers(self, df: pd.DataFrame) -> None:
        """Fit a per-feature StandardScaler on ``df`` (typically the train slice).

        Kept separate from transforming so the scaler can be fit on the
        in-sample train segment and then applied to held-out segments without
        leaking test-period statistics into training.
        """
        self.scalers = {}
        for col in self._feature_cols(df):
            scaler = StandardScaler()
            scaler.fit(df[[col]].fillna(0))
            self.scalers[col] = scaler

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously-fit scalers to ``df`` (raises if none were fit)."""
        if not self.scalers:
            raise RuntimeError(
                "transform_features called before fit_scalers; no scalers "
                "available. Call fit_scalers() / normalize_features(fit_scaler=True) "
                "on the train segment first.")
        df = df.copy()
        for col in self._feature_cols(df):
            if col in self.scalers:
                df[col] = self.scalers[col].transform(df[[col]].fillna(0))
        return df

    def normalize_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Fit-and-transform (``fit_scaler=True``) or transform-only.

        Backwards-compatible wrapper around :meth:`fit_scalers` /
        :meth:`transform_features`.
        """
        if fit_scaler:
            self.fit_scalers(df)
        return self.transform_features(df)

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