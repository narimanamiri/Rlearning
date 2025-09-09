import yfinance as yf
import ccxt
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Union

class DataFetcher:
    def __init__(self, config):
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
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                data_dict[symbol] = df
                print(f"✅ Successfully fetched {symbol}")
            except Exception as e:
                print(f"❌ Failed to fetch {symbol}: {e}. Skipping...")
                continue
        return data_dict