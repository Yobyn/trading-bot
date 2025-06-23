import ccxt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from loguru import logger
from config import config

class DataFetcher:
    def __init__(self, exchange_name: str = None, api_key: str = None, secret: str = None):
        self.exchange_name = exchange_name or config.exchange
        self.api_key = api_key or config.exchange_api_key
        self.secret = secret or config.exchange_secret
        
        # Initialize exchange
        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.secret,
            'sandbox': config.paper_trading,
            'enableRateLimit': True,
        })
        
        # Load markets
        try:
            self.exchange.load_markets()
            logger.info(f"Connected to {self.exchange_name}")
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_name}: {e}")
    
    async def get_market_data(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """Get comprehensive market data including technical indicators"""
        symbol = symbol or config.symbol
        timeframe = timeframe or config.timeframe
        
        try:
            # Fetch OHLCV data
            ohlcv = await self._fetch_ohlcv(symbol, timeframe, limit=100)
            if not ohlcv:
                raise Exception(f"Failed to fetch OHLCV data for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Get current ticker
            ticker = await self._fetch_ticker(symbol)
            
            # Get current position (placeholder - would be implemented in trading engine)
            current_position = await self._get_current_position(symbol)
            
            # Get portfolio value (placeholder - would be implemented in trading engine)
            portfolio_value = await self._get_portfolio_value()
            
            return {
                'symbol': symbol,
                'current_price': ticker.get('last', df['close'].iloc[-1]),
                'price_change_24h': ticker.get('percentage', 0),
                'volume_24h': ticker.get('quoteVolume', 0),
                'high_24h': ticker.get('high', 0),
                'low_24h': ticker.get('low', 0),
                'timestamp': datetime.now().isoformat(),
                'current_position': current_position,
                'portfolio_value': portfolio_value,
                **indicators
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return self._get_default_market_data(symbol)
    
    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List:
        """Fetch OHLCV data from exchange"""
        try:
            # Use sync method for now (ccxt doesn't have async OHLCV)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return []
    
    async def _fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker data"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return {}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            close_prices = df['close']
            
            # RSI
            rsi = self._calculate_rsi(close_prices, period=14)
            
            # MACD
            macd, signal, histogram = self._calculate_macd(close_prices)
            
            # Moving Averages
            ma_20 = self._calculate_sma(close_prices, period=20)
            ma_50 = self._calculate_sma(close_prices, period=50)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices, period=20)
            
            # Volume indicators
            volume_sma = self._calculate_sma(df['volume'], period=20)
            
            return {
                'rsi': float(rsi.iloc[-1]) if not rsi.empty else 50,
                'macd': float(macd.iloc[-1]) if not macd.empty else 0,
                'macd_signal': float(signal.iloc[-1]) if not signal.empty else 0,
                'macd_histogram': float(histogram.iloc[-1]) if not histogram.empty else 0,
                'ma_20': float(ma_20.iloc[-1]) if not ma_20.empty else close_prices.iloc[-1],
                'ma_50': float(ma_50.iloc[-1]) if not ma_50.empty else close_prices.iloc[-1],
                'bb_upper': float(bb_upper.iloc[-1]) if not bb_upper.empty else close_prices.iloc[-1],
                'bb_middle': float(bb_middle.iloc[-1]) if not bb_middle.empty else close_prices.iloc[-1],
                'bb_lower': float(bb_lower.iloc[-1]) if not bb_lower.empty else close_prices.iloc[-1],
                'volume_sma': float(volume_sma.iloc[-1]) if not volume_sma.empty else df['volume'].iloc[-1],
                'price_above_ma20': close_prices.iloc[-1] > ma_20.iloc[-1] if not ma_20.empty else False,
                'price_above_ma50': close_prices.iloc[-1] > ma_50.iloc[-1] if not ma_50.empty else False,
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return self._get_default_indicators()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        sma = self._calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    async def _get_current_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position for the symbol (placeholder)"""
        # This would be implemented in the trading engine
        return {
            'side': 'None',
            'size': 0,
            'entry_price': 0,
            'unrealized_pnl': 0
        }
    
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value (placeholder)"""
        # This would be implemented in the trading engine
        return 10000.0  # Default portfolio value
    
    def _get_default_market_data(self, symbol: str) -> Dict[str, Any]:
        """Return default market data when fetching fails"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'price_change_24h': 0,
            'volume_24h': 0,
            'high_24h': 0,
            'low_24h': 0,
            'timestamp': datetime.now().isoformat(),
            'current_position': {'side': 'None', 'size': 0, 'entry_price': 0, 'unrealized_pnl': 0},
            'portfolio_value': 10000.0,
            **self._get_default_indicators()
        }
    
    def _get_default_indicators(self) -> Dict[str, Any]:
        """Return default indicator values"""
        return {
            'rsi': 50,
            'macd': 0,
            'macd_signal': 0,
            'macd_histogram': 0,
            'ma_20': 0,
            'ma_50': 0,
            'bb_upper': 0,
            'bb_middle': 0,
            'bb_lower': 0,
            'volume_sma': 0,
            'price_above_ma20': False,
            'price_above_ma50': False,
        }