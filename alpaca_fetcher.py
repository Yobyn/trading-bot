import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from loguru import logger
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import config

class AlpacaDataFetcher:
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = api_key or config.exchange_api_key
        self.secret_key = secret_key or config.exchange_secret
        self.paper = paper
        
        # Initialize Alpaca clients
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        
        logger.info(f"Connected to Alpaca {'Paper' if self.paper else 'Live'} Trading")
    
    async def get_market_data(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """Get comprehensive market data including technical indicators"""
        symbol = symbol or config.symbol.replace('/', '')  # Convert BTC/USDT to BTCUSDT
        timeframe = timeframe or config.timeframe
        
        try:
            # Convert timeframe to Alpaca format
            alpaca_timeframe = self._convert_timeframe(timeframe)
            
            # Fetch historical data
            bars = await self._fetch_bars(symbol, alpaca_timeframe, limit=100)
            if bars is None or bars.empty:
                raise Exception(f"Failed to fetch data for {symbol}")
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(bars)
            
            # Get current price (last bar)
            current_price = bars['close'].iloc[-1]
            
            # Get current position
            current_position = await self._get_current_position(symbol)
            
            # Get portfolio value
            portfolio_value = await self._get_portfolio_value()
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_24h': self._calculate_24h_change(bars),
                'volume_24h': bars['volume'].iloc[-1] if len(bars) > 0 else 0,
                'high_24h': bars['high'].iloc[-1] if len(bars) > 0 else current_price,
                'low_24h': bars['low'].iloc[-1] if len(bars) > 0 else current_price,
                'timestamp': datetime.now().isoformat(),
                'current_position': current_position,
                'portfolio_value': portfolio_value,
                **indicators
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return self._get_default_market_data(symbol)
    
    async def _fetch_bars(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> pd.DataFrame:
        """Fetch historical bars from Alpaca"""
        try:
            # For free tier, use daily data and get more historical data
            if timeframe == TimeFrame.Hour:
                timeframe = TimeFrame.Day
                limit = min(limit, 30)  # Limit to 30 days for free tier
            
            # Calculate start time (limit bars ago)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=limit * self._timeframe_to_days(timeframe))
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_time,
                end=end_time
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            if bars and hasattr(bars, 'df'):
                df = bars.df
                if symbol in df.index.get_level_values(1):
                    return df.xs(symbol, level=1)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching bars: {e}")
            return pd.DataFrame()
    
    def _convert_timeframe(self, timeframe: str) -> TimeFrame:
        """Convert string timeframe to Alpaca TimeFrame"""
        mapping = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame.Minute,  # Alpaca doesn't have 5m, use 1m
            '15m': TimeFrame.Minute,  # Alpaca doesn't have 15m, use 1m
            '30m': TimeFrame.Minute,  # Alpaca doesn't have 30m, use 1m
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day,
            '1w': TimeFrame.Week,
            '1M': TimeFrame.Month
        }
        return mapping.get(timeframe, TimeFrame.Hour)
    
    def _timeframe_to_days(self, timeframe: TimeFrame) -> int:
        """Convert TimeFrame to approximate days"""
        mapping = {
            TimeFrame.Minute: 1/1440,  # 1 minute
            TimeFrame.Hour: 1/24,
            TimeFrame.Day: 1,
            TimeFrame.Week: 7,
            TimeFrame.Month: 30
        }
        return mapping.get(timeframe, 1)
    
    def _calculate_24h_change(self, bars: pd.DataFrame) -> float:
        """Calculate 24-hour price change percentage"""
        if len(bars) < 2:
            return 0.0
        
        current_price = bars['close'].iloc[-1]
        prev_price = bars['close'].iloc[-2]
        
        if prev_price == 0:
            return 0.0
        
        return ((current_price - prev_price) / prev_price) * 100
    
    def _calculate_indicators(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            close_prices = bars['close']
            
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
            volume_sma = self._calculate_sma(bars['volume'], period=20)
            
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
                'volume_sma': float(volume_sma.iloc[-1]) if not volume_sma.empty else bars['volume'].iloc[-1],
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
        """Get current position for the symbol"""
        try:
            positions = self.trading_client.get_all_positions()
            for position in positions:
                if position.symbol == symbol:
                    return {
                        'side': 'long' if position.qty > 0 else 'short',
                        'size': abs(float(position.qty)),
                        'entry_price': float(position.avg_entry_price),
                        'unrealized_pnl': float(position.unrealized_pl)
                    }
            return {'side': 'None', 'size': 0, 'entry_price': 0, 'unrealized_pnl': 0}
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return {'side': 'None', 'size': 0, 'entry_price': 0, 'unrealized_pnl': 0}
    
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            account = self.trading_client.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 10000.0  # Default value
    
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