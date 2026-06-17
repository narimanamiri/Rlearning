import yfinance as yf
import ccxt
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Union

class DataFetcher:
    def __init__(self, config):
        self.config = config
        self.crypto_exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
            'retries': 3,
            'options': {'defaultType': 'spot'}
        })
        # Optional proxy — uncomment if needed
        # self.crypto_exchange.proxies = {
        #     'http': 'http://proxy:port',
        #     'https': 'http://proxy:port',
        # }

    def fetch_ohlcv(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        STOCK_TICKERS = {"AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "SPY", "QQQ", "BTC-USD", "ETH-USD"}
        
        if symbol in STOCK_TICKERS or symbol.endswith(('-USD', '-EUR', '-JPY')):
            return self._fetch_stock(symbol, interval, period)
        else:
            return self._fetch_crypto(symbol, interval)

    def _fetch_stock(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(yf.download, symbol, period=period, interval=interval)
            try:
                data = future.result(timeout=30)
                if data.empty:
                    raise ValueError(f"No data for {symbol}")
                return data
            except FuturesTimeoutError:
                raise ValueError(f"Timeout fetching {symbol} from Yahoo Finance")

    def _fetch_crypto(self, symbol: str, interval: str) -> pd.DataFrame:
        symbol_pair = f"{symbol}/USDT"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ohlcv = self.crypto_exchange.fetch_ohlcv(symbol_pair, interval)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to fetch {symbol} after {max_retries} attempts: {e}")
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed for {symbol}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLCV columns to lowercase regardless of the source.

        yfinance returns capitalized columns (and sometimes a MultiIndex when
        a single ticker is downloaded), while ccxt returns lowercase columns.
        This collapses both to the canonical ``open/high/low/close/volume``.
        """
        df = df.copy()
        # Flatten a possible MultiIndex (yfinance) by keeping the price field.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected OHLCV columns: {missing}")
        return df[required]

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        for symbol in self.config['data']['symbols']:
            print(f"Fetching {symbol}...")
            try:
                df = self.fetch_ohlcv(
                    symbol,
                    self.config['data']['interval'],
                    self.config['data']['period']
                )
                df = self._standardize_columns(df)
                data_dict[symbol] = df
                print(f"✅ Successfully fetched {symbol}")
            except Exception as e:
                print(f"❌ Failed to fetch {symbol}: {e}. Skipping...")
                continue
        return data_dict