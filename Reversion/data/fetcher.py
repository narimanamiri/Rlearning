import yfinance as yf
import ccxt
import pandas as pd
import os
from typing import List, Dict, Union

class DataFetcher:
    def __init__(self, config):
        self.config = config
        self.crypto_exchange = ccxt.binance({'enableRateLimit': True})

    def fetch_ohlcv(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        if '-' in symbol or symbol.endswith(('USD', 'EUR', 'JPY')):
            return self._fetch_stock(symbol, interval, period)
        else:
            return self._fetch_crypto(symbol, interval)

    def _fetch_stock(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data for {symbol}")
        return data

    def _fetch_crypto(self, symbol: str, interval: str) -> pd.DataFrame:
        symbol_pair = f"{symbol}/USDT"
        try:
            ohlcv = self.crypto_exchange.fetch_ohlcv(symbol_pair, interval)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            raise ValueError(f"Failed to fetch {symbol}: {e}")

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        for symbol in self.config['data']['symbols']:
            print(f"Fetching {symbol}...")
            df = self.fetch_ohlcv(
                symbol,
                self.config['data']['interval'],
                self.config['data']['period']
            )
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            data_dict[symbol] = df
        return data_dict