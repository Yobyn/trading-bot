#!/usr/bin/env python3
"""
Coinbase Smart Allocation Bot (Advanced API)
Automatically invests available capital into the top 3 positions with the lowest price relative to their weekly average
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from loguru import logger
import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
# Define ONE_HOUR constant for candle granularity (3600 seconds)
ONE_HOUR = "ONE_HOUR"
import asyncio
import uuid

from config import config
from llm_client import LLMClient
from asset_config import ASSET_PORTFOLIOS

load_dotenv()

# Ensure analysis folder exists
ANALYSIS_FOLDER = "analysis"
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

class CoinbaseDataFetcher:
    def __init__(self):
        api_key = os.getenv('COINBASE_API_KEY')
        api_secret_file = os.getenv('COINBASE_API_SECRET_FILE')
        
        if not api_key or not api_secret_file:
            raise ValueError("COINBASE_API_KEY and COINBASE_API_SECRET_FILE must be set in .env")
        
        with open(api_secret_file, 'r') as f:
            api_secret = f.read()
        
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        logger.info("Connected to Coinbase Advanced API (DataFetcher)")
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            # Coinbase Advanced API uses '-' instead of '/' in symbols
            product_id = symbol.replace('/', '-')
            # Get product book (this gives us correct real-time prices)
            book = self.client.get_product_book(product_id=product_id)
            
            # Extract price from the product book
            if book.pricebook and book.pricebook.bids and book.pricebook.asks:
                pricebook = book.pricebook
                if pricebook.bids and len(pricebook.bids) > 0:
                    best_bid = float(pricebook.bids[0].price)
                else:
                    best_bid = 0
                if pricebook.asks and len(pricebook.asks) > 0:
                    best_ask = float(pricebook.asks[0].price)
                else:
                    best_ask = 0
                
                # Use mid-price as reference, but return both bid and ask for proper usage
                if best_bid > 0 and best_ask > 0:
                    current_price = (best_bid + best_ask) / 2  # Mid-price for reference
                elif best_ask > 0:
                    current_price = best_ask  # Fallback to ask if no bid
                elif best_bid > 0:
                    current_price = best_bid  # Fallback to bid if no ask
                else:
                    current_price = 0
            else:
                best_bid = 0
                best_ask = 0
                current_price = 0
            
            # Get historical candles for 3-month average (90 days)
            now = datetime.now(timezone.utc)
            start_dt = now - timedelta(days=90)  # 3 months of historical data for more recent trends
            start = int(start_dt.timestamp())
            end = int(now.timestamp())
            
            # Try to get daily candles for ~11 months (more efficient than hourly)
            try:
                candles = self.client.get_candles(product_id=product_id, start=str(start), end=str(end), granularity='ONE_DAY')
                
                # Extract candles with timestamps and sort by timestamp to ensure chronological order
                candle_data = []
                if candles and candles.candles:
                    for candle in candles.candles:
                        if candle.close is not None:
                            # Use the correct timestamp attribute for Coinbase candles
                            timestamp = getattr(candle, 'start', getattr(candle, 'timestamp', 0))
                            candle_data.append({
                                'timestamp': int(timestamp),
                                'close': float(candle.close)
                            })
                
                # Sort by timestamp (oldest first, newest last)
                candle_data.sort(key=lambda x: x['timestamp'])
                closes = [candle['close'] for candle in candle_data]
                
                # Validate we have at least 60 days of data (allowing for some missing days and weekends)
                if len(closes) < 60:
                    logger.warning(f"{symbol}: Only {len(closes)} days of historical data available (need 60+). Skipping this asset.")
                    return None  # Skip assets without sufficient historical data
                
                three_month_avg = np.mean(closes) if closes else 0
                
                # Calculate weekly average (last 7 days) with validation
                if len(closes) >= 7:
                    weekly_closes = closes[-7:]
                    weekly_avg = np.mean(weekly_closes)
                    
                    # Debug logging to verify weekly average calculation
                    if len(candle_data) >= 7:
                        recent_timestamps = [candle_data[i]['timestamp'] for i in range(-7, 0)]
                        recent_dates = [datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d') for ts in recent_timestamps]
                        logger.debug(f"{symbol}: Weekly avg €{weekly_avg:.6f} from dates {recent_dates} (prices: {[f'{p:.6f}' for p in weekly_closes]})")
                        
                        # Sanity check: verify dates are actually recent (within last 10 days)
                        most_recent_timestamp = recent_timestamps[-1]
                        days_ago = (datetime.now(timezone.utc).timestamp() - most_recent_timestamp) / 86400
                        if days_ago > 10:
                            logger.warning(f"{symbol}: Weekly average using data from {days_ago:.1f} days ago! Using current price instead.")
                            weekly_avg = current_price
                else:
                    weekly_avg = three_month_avg
                    
                logger.info(f"{symbol}: Using {len(closes)} days of historical data for ~3-month average, weekly avg from last 7 days")
                
            except Exception as candle_error:
                logger.warning(f"Failed to get daily candles for {symbol}, trying hourly: {candle_error}")
                # Fallback to hourly candles for last 30 days if daily fails
                start_dt = now - timedelta(days=30)
                start = int(start_dt.timestamp())
                candles = self.client.get_candles(product_id=product_id, start=str(start), end=str(end), granularity='ONE_HOUR')
                
                # Extract hourly candles with timestamps and sort by timestamp
                candle_data = []
                if candles and candles.candles:
                    for candle in candles.candles:
                        if candle.close is not None:
                            timestamp = getattr(candle, 'start', getattr(candle, 'timestamp', 0))
                            candle_data.append({
                                'timestamp': int(timestamp),
                                'close': float(candle.close)
                            })
                
                # Sort by timestamp (oldest first, newest last)
                candle_data.sort(key=lambda x: x['timestamp'])
                closes = [candle['close'] for candle in candle_data]
                
                if len(closes) < 600:  # Less than ~25 days of hourly data
                    logger.warning(f"{symbol}: Insufficient historical data even with hourly fallback. Skipping this asset.")
                    return None
                
                three_month_avg = np.mean(closes) if closes else 0
                
                # Calculate weekly average (last 7 days * 24 hours = 168 hours) with validation
                if len(closes) >= 168:
                    weekly_closes = closes[-168:]
                    weekly_avg = np.mean(weekly_closes)
                    
                    # Debug logging for hourly data
                    if len(candle_data) >= 168:
                        most_recent_timestamp = candle_data[-1]['timestamp']
                        days_ago = (datetime.now(timezone.utc).timestamp() - most_recent_timestamp) / 86400
                        logger.debug(f"{symbol}: Hourly weekly avg €{weekly_avg:.6f} from {len(weekly_closes)} hours (most recent: {days_ago:.1f} days ago)")
                        
                        # Sanity check for hourly data
                        if days_ago > 10:
                            logger.warning(f"{symbol}: Hourly weekly average using data from {days_ago:.1f} days ago! Using current price instead.")
                            weekly_avg = current_price
                else:
                    weekly_avg = three_month_avg
                    
                logger.warning(f"{symbol}: Using fallback 30-day average instead of ~3-month average (insufficient data)")
            
            # If current price seems wrong (too low compared to 3-month average), use most recent candle close
            if three_month_avg > 0 and current_price > 0 and current_price < (three_month_avg * 0.1):  # Current price is less than 10% of 3-month average
                logger.warning(f"Current price {current_price} seems too low vs 3-month avg {three_month_avg}. Using latest candle close.")
                if closes:
                    corrected_price = float(closes[-1])  # Convert to regular Python float
                    # Sanity check: corrected price should be within reasonable range of 3-month average
                    if corrected_price > 0 and corrected_price < (three_month_avg * 10):  # Less than 10x 3-month average
                        current_price = corrected_price
                        logger.info(f"Updated current price to: {current_price}")
                    else:
                        logger.warning(f"Corrected price {corrected_price} seems unreasonable vs 3-month avg {three_month_avg}. Using 3-month average.")
                        current_price = float(three_month_avg)
                else:
                    logger.warning("No candle data available. Using 3-month average.")
                    current_price = float(three_month_avg)
            
            # Calculate technical indicators from historical data
            indicators = self._calculate_technical_indicators(closes) if closes else {}
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'three_month_average': three_month_avg,
                'weekly_average': weekly_avg,
                'bid': best_bid,
                'ask': best_ask,
                'volume': 0,  # Not available in basic ticker
                # Technical indicators
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', 0),
                'macd_signal': indicators.get('macd_signal', 0),
                'macd_histogram': indicators.get('macd_histogram', 0),
                'ma_20': indicators.get('ma_20', current_price),
                'ma_50': indicators.get('ma_50', current_price),
                'bb_upper': indicators.get('bb_upper', current_price),
                'bb_middle': indicators.get('bb_middle', current_price),
                'bb_lower': indicators.get('bb_lower', current_price),
                'price_above_ma20': current_price > indicators.get('ma_20', current_price),
                'price_above_ma50': current_price > indicators.get('ma_50', current_price),
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            
            # Retry logic for network issues
            max_retries = 2
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Retrying market data fetch for {symbol} (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(retry_delay)
                    
                    # Retry the main data fetching logic
                    ticker = self.client.get_product_book(product_id=product_id, limit=1)
                    if not ticker or not hasattr(ticker, 'pricebook') or not ticker.pricebook:
                        continue
                        
                    pricebook = ticker.pricebook
                    bid_price = float(pricebook.bids[0].price) if pricebook.bids and len(pricebook.bids) > 0 else None
                    ask_price = float(pricebook.asks[0].price) if pricebook.asks and len(pricebook.asks) > 0 else None
                    current_price = (bid_price + ask_price) / 2 if bid_price and ask_price else None
                    
                    if not current_price:
                        continue
                    
                    # Get historical data for yearly average and technical indicators
                    try:
                        # Try ~9-month data first (270 days to stay well under 350 candle limit)
                        candles = self.client.get_product_candles(
                            product_id=product_id,
                            start=int((datetime.now() - timedelta(days=270)).timestamp()),
                            end=int(datetime.now().timestamp()),
                            granularity='ONE_DAY'
                        )
                        
                        if candles and candles.get('candles') and len(candles['candles']) >= 250:
                            prices = [float(candle['close']) for candle in candles['candles']]
                            yearly_average = sum(prices) / len(prices) if prices else current_price
                            weekly_average = sum(prices[-7:]) / len(prices[-7:]) if len(prices) >= 7 else yearly_average
                            indicators = self._calculate_technical_indicators(prices)
                        else:
                            # Fallback to 30-day average if insufficient yearly data
                            candles = self.client.get_product_candles(
                                product_id=product_id,
                                start=int((datetime.now() - timedelta(days=30)).timestamp()),
                                end=int(datetime.now().timestamp()),
                                granularity=ONE_HOUR
                            )
                            if candles and candles.get('candles') and len(candles['candles']) >= 600:
                                prices = [float(candle['close']) for candle in candles['candles']]
                                yearly_average = sum(prices) / len(prices) if prices else current_price
                                weekly_average = sum(prices[-168:]) / len(prices[-168:]) if len(prices) >= 168 else yearly_average
                                indicators = self._calculate_technical_indicators(prices)
                            else:
                                # Skip asset if insufficient data
                                continue
                    except Exception:
                        yearly_average = current_price
                        weekly_average = current_price
                        indicators = {}
                    
                    # Success! Return the data
                    return {
                        'symbol': symbol,
                        'current_price': current_price,
                        'bid': bid_price,
                        'ask': ask_price,
                        'yearly_average': yearly_average,
                        'weekly_average': weekly_average,
                        'volume': 0,  # Volume not easily available from ticker
                        'rsi': indicators.get('rsi', 50),
                        'macd': indicators.get('macd', 0),
                        'ma_20': indicators.get('ma_20', current_price),
                        'ma_50': indicators.get('ma_50', current_price),
                        'bb_upper': indicators.get('bb_upper', current_price),
                        'bb_middle': indicators.get('bb_middle', current_price),
                        'bb_lower': indicators.get('bb_lower', current_price),
                        'price_above_ma20': current_price > indicators.get('ma_20', current_price),
                        'price_above_ma50': current_price > indicators.get('ma_50', current_price),
                    }
                    
                except Exception as retry_e:
                    logger.warning(f"Retry {attempt + 1} failed for {symbol}: {retry_e}")
                    if attempt < max_retries - 1:
                        retry_delay *= 2  # Exponential backoff
                    continue
            
            logger.error(f"All retry attempts failed for {symbol}")
            return None
    
    def _calculate_technical_indicators(self, prices: List[float]) -> Dict[str, float]:
        """Calculate technical indicators from price data"""
        if len(prices) < 50:  # Need at least 50 data points for reliable indicators
            return {}
        
        try:
            import pandas as pd
            
            # Convert to pandas Series for easier calculation
            price_series = pd.Series(prices)
            
            # RSI calculation
            rsi = self._calculate_rsi(price_series, period=14)
            
            # MACD calculation
            macd, signal, histogram = self._calculate_macd(price_series)
            
            # Moving averages
            ma_20 = price_series.rolling(window=20).mean().iloc[-1]
            ma_50 = price_series.rolling(window=50).mean().iloc[-1]
            
            # Bollinger Bands
            bb_middle = ma_20
            bb_std = price_series.rolling(window=20).std().iloc[-1]
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            return {
                'rsi': float(rsi.iloc[-1]) if not rsi.empty else 50,
                'macd': float(macd.iloc[-1]) if not macd.empty else 0,
                'macd_signal': float(signal.iloc[-1]) if not signal.empty else 0,
                'macd_histogram': float(histogram.iloc[-1]) if not histogram.empty else 0,
                'ma_20': float(ma_20) if pd.notna(ma_20) else prices[-1],
                'ma_50': float(ma_50) if pd.notna(ma_50) else prices[-1],
                'bb_upper': float(bb_upper) if pd.notna(bb_upper) else prices[-1],
                'bb_middle': float(bb_middle) if pd.notna(bb_middle) else prices[-1],
                'bb_lower': float(bb_lower) if pd.notna(bb_lower) else prices[-1],
            }
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> tuple:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal_period).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

class CoinbaseTradingEngine:
    def __init__(self):
        api_key = os.getenv('COINBASE_API_KEY')
        api_secret_file = os.getenv('COINBASE_API_SECRET_FILE')
        
        if not api_key or not api_secret_file:
            raise ValueError("COINBASE_API_KEY and COINBASE_API_SECRET_FILE must be set in .env")
        
        with open(api_secret_file, 'r') as f:
            api_secret = f.read()
        
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        
        # Paper trading balance (simulated funds)
        self.paper_balance = 10000.0  # Start with €10,000 EUR
        self.paper_positions = {}  # Track paper trading positions
        
        # Coinbase precision requirements for different assets
        # Based on actual testing and Coinbase API errors, these are the working precision rules
        self.precision_rules = {
            'BTC-EUR': {'size_decimals': 8, 'price_decimals': 2},
            'ETH-EUR': {'size_decimals': 6, 'price_decimals': 2},
            'SOL-EUR': {'size_decimals': 4, 'price_decimals': 2},  # SOL supports 4 decimal places for size
            'ADA-EUR': {'size_decimals': 4, 'price_decimals': 4},  # ADA supports 4 decimal places for size
            # Default fallback
            'default': {'size_decimals': 4, 'price_decimals': 2}  # Most cryptos support 4 decimal places
        }
        
        logger.info("Connected to Coinbase Advanced API (TradingEngine)")
        if not config.trading_enabled:
            logger.info(f"📊 Paper Trading Balance: €{self.paper_balance:.2f} EUR")
    
    def _get_sell_price_for_balance(self, market_data: Dict[str, Any]) -> float:
        """Get the price to use for portfolio valuation (BID price)"""
        bid_price = market_data.get('bid', 0)
        current_price = market_data.get('current_price', 0)
        weekly_avg = market_data.get('weekly_average', current_price)
        
        # Sanity check: if bid price seems unreasonable compared to current price, use current price
        if bid_price > 0 and current_price > 0 and weekly_avg > 0:
            # If bid is less than 50% of weekly average, it's probably wrong
            if bid_price < (weekly_avg * 0.5):
                logger.warning(f"Bid price {bid_price} seems too low vs weekly avg {weekly_avg}. Using current price for portfolio valuation.")
                return current_price
            return bid_price
        # Fallback to current price if bid not available or unreasonable
        return current_price
    
    def get_account_balance(self) -> float:
        """Get total portfolio value in EUR (cash + crypto holdings) with retry logic"""
        if not config.trading_enabled:
            return self.paper_balance
        
        # Retry logic for network issues
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                accounts = self.client.get_accounts()
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error getting account balance (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to get account balance after {max_retries} attempts: {e}")
                    return 0.0  # Return 0 on final failure
        
        try:
            total_balance = 0.0
            
            for account in accounts['accounts'] if isinstance(accounts, dict) else accounts.accounts:
                # Always handle as dict for robustness
                if isinstance(account, dict):
                    currency = account.get('currency', '')
                    available_balance = account.get('available_balance', {})
                    value = available_balance.get('value', 0)
                else:
                    currency = getattr(account, 'currency', '')
                    available_balance = getattr(account, 'available_balance', {})
                    value = available_balance.get('value', 0)
                
                if currency == 'EUR':
                    try:
                        float_value = float(value)
                        total_balance += float_value
                    except Exception:
                        logger.warning(f"Could not convert EUR balance value: {value}")
                elif currency in ['BTC', 'ETH', 'SOL', 'ADA', 'LTC', 'BCH', 'XRP', 'DOGE', 'ETC', 'UNI', 'AAVE', 'CRV', 'SNX', '1INCH', 'ENS', 'DOT', 'AVAX', 'ATOM', 'ALGO', 'MINA', 'XTZ', 'LINK', 'BAT', 'CHZ', 'MANA', 'FIL', 'GRT', 'ICP', 'MASK', 'XLM', 'EOS', 'APE', 'AXS', 'RNDR', 'MATIC', 'ANKR', 'SHIB', 'USDC', 'USDT', 'CGLD', 'CRO']:
                    # Convert crypto holdings to EUR value
                    try:
                        crypto_amount = float(value)
                        if crypto_amount > 0:
                            # Get current price in EUR from data fetcher
                            # Note: This creates a temporary data fetcher instance
                            temp_data_fetcher = CoinbaseDataFetcher()
                            symbol = f"{currency}/EUR"
                            market_data = temp_data_fetcher.get_market_data(symbol)
                            if market_data and market_data.get('current_price', 0) > 0:
                                # Use corrected current price for portfolio valuation (already corrected in get_market_data)
                                # The current_price from get_market_data is already the corrected price
                                sell_price = market_data['current_price']
                                crypto_eur_value = crypto_amount * sell_price
                                total_balance += crypto_eur_value
                            else:
                                logger.warning(f"Could not get EUR price for {currency}")
                    except Exception as e:
                        logger.warning(f"Could not convert {currency} balance value: {value}, error: {e}")
            
            return total_balance
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0


    
    def set_paper_balance(self, amount: float):
        """Set paper trading balance (for testing)"""
        if not config.trading_enabled:
            self.paper_balance = amount
            logger.info(f"📊 Paper Trading Balance set to: €{self.paper_balance:.2f} EUR")
        else:
            logger.warning("Cannot set paper balance when live trading is enabled")
    
    def get_paper_positions(self) -> Dict[str, float]:
        """Get current paper trading positions"""
        return self.paper_positions.copy()
    
    def format_order_params(self, product_id: str, amount: float, price: Optional[float] = None) -> tuple:
        """Format amount and price according to Coinbase precision requirements"""
        rules = self.precision_rules.get(product_id, self.precision_rules['default'])
        
        logger.info(f"🔧 PRECISION FORMATTING for {product_id}:")
        logger.info(f"  Precision Rules: {rules}")
        logger.info(f"  Original Amount: {amount}")
        logger.info(f"  Original Price: {price}")
        
        # Format amount (base_size) with proper decimal places
        formatted_amount = round(amount, rules['size_decimals'])
        formatted_amount_str = f"{formatted_amount:.{rules['size_decimals']}f}".rstrip('0').rstrip('.')
        
        logger.info(f"  Amount Formatting:")
        logger.info(f"    Rounded to {rules['size_decimals']} decimals: {formatted_amount}")
        logger.info(f"    Formatted string: '{formatted_amount_str}'")
        
        # Format price with proper decimal places if provided
        formatted_price_str = None
        if price is not None:
            formatted_price = round(price, rules['price_decimals'])
            formatted_price_str = f"{formatted_price:.{rules['price_decimals']}f}".rstrip('0').rstrip('.')
            
            logger.info(f"  Price Formatting:")
            logger.info(f"    Rounded to {rules['price_decimals']} decimals: {formatted_price}")
            logger.info(f"    Formatted string: '{formatted_price_str}'")
        else:
            logger.info(f"  Price: None (market order)")
        
        logger.info(f"  Final Result: amount='{formatted_amount_str}', price='{formatted_price_str}'")
        return formatted_amount_str, formatted_price_str
    
    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> bool:
        try:
            if not config.trading_enabled:
                # Paper trading - update balance and positions
                order_value = amount * (price or 1.0)
                
                if side.upper() == 'BUY':
                    if order_value > self.paper_balance:
                        logger.warning(f"PAPER TRADE: Insufficient balance (€{self.paper_balance:.2f}) for €{order_value:.2f} buy order")
                        return False
                    
                    self.paper_balance -= order_value
                    if symbol in self.paper_positions:
                        self.paper_positions[symbol] += amount
                    else:
                        self.paper_positions[symbol] = amount
                    
                    logger.info(f"PAPER TRADE: BUY {amount} {symbol} at €{price or 'market'} = €{order_value:.2f}")
                    logger.info(f"📊 Paper Balance: €{self.paper_balance:.2f} EUR")
                
                elif side.upper() == 'SELL':
                    if symbol not in self.paper_positions or self.paper_positions[symbol] < amount:
                        logger.warning(f"PAPER TRADE: Insufficient {symbol} position for sell order")
                        return False
                    
                    self.paper_balance += order_value
                    self.paper_positions[symbol] -= amount
                    if self.paper_positions[symbol] <= 0:
                        del self.paper_positions[symbol]
                    
                    logger.info(f"PAPER TRADE: SELL {amount} {symbol} at €{price or 'market'} = €{order_value:.2f}")
                    logger.info(f"📊 Paper Balance: €{self.paper_balance:.2f} EUR")
                
                return True
            
            # Convert symbol format
            product_id = symbol.replace('/', '-')
            client_order_id = str(uuid.uuid4())
            
            # Format order parameters with proper precision
            formatted_amount, formatted_price = self.format_order_params(product_id, amount, price)
            
            # Log detailed request information
            logger.info(f"📤 COINBASE API REQUEST:")
            logger.info(f"  Original Input: symbol={symbol}, side={side}, amount={amount}, price={price}")
            logger.info(f"  Converted: product_id={product_id}, formatted_amount={formatted_amount}, formatted_price={formatted_price}")
            logger.info(f"  Client Order ID: {client_order_id}")
            
            # Place order using Coinbase Advanced API
            if price:
                # Limit order
                order_config = {
                    'limit_limit_gtc': {
                        'base_size': formatted_amount,
                        'limit_price': formatted_price
                    }
                }
                logger.info(f"  Order Type: LIMIT")
                logger.info(f"  Order Configuration: {order_config}")
                
                order = self.client.create_order(
                    product_id=product_id,
                    side=side.upper(),
                    client_order_id=client_order_id,
                    order_configuration=order_config
                )
            else:
                # Market order
                if side.upper() == 'BUY':
                    # For market BUY orders, use quote_size (EUR amount to spend)
                    # EUR amounts need to be formatted to exactly 2 decimal places for Coinbase
                    eur_amount = round(float(formatted_amount), 2)
                    eur_amount_str = f"{eur_amount:.2f}"
                    
                    logger.info(f"  Market BUY EUR Formatting:")
                    logger.info(f"    Original formatted_amount: '{formatted_amount}'")
                    logger.info(f"    Rounded EUR amount: {eur_amount}")
                    logger.info(f"    Final EUR string: '{eur_amount_str}'")
                    
                    order_config = {
                        'market_market_ioc': {
                            'quote_size': eur_amount_str
                        }
                    }
                    logger.info(f"  Order Type: MARKET BUY")
                    logger.info(f"  Order Configuration: {order_config}")
                    
                    order = self.client.create_order(
                        product_id=product_id,
                        side=side.upper(),
                        client_order_id=client_order_id,
                        order_configuration=order_config
                    )
                else:
                    # For market SELL orders, use base_size (crypto amount to sell)
                    order_config = {
                        'market_market_ioc': {
                            'base_size': formatted_amount
                        }
                    }
                    logger.info(f"  Order Type: MARKET SELL")
                    logger.info(f"  Order Configuration: {order_config}")
                    
                    order = self.client.create_order(
                        product_id=product_id,
                        side=side.upper(),
                        client_order_id=client_order_id,
                        order_configuration=order_config
                    )
            
            # Log detailed response information
            logger.info(f"📥 COINBASE API RESPONSE:")
            logger.info(f"  Response Type: {type(order)}")
            
            if order:
                # Convert response to dict for comprehensive logging
                if hasattr(order, '__dict__'):
                    response_dict = order.__dict__
                else:
                    response_dict = str(order)
                logger.info(f"  Full Response: {response_dict}")
                
                # Log specific response attributes
                if hasattr(order, 'success'):
                    logger.info(f"  Success: {order.success}")
                if hasattr(order, 'success_response'):
                    logger.info(f"  Success Response: {order.success_response}")
                if hasattr(order, 'error_response'):
                    logger.info(f"  Error Response: {order.error_response}")
                if hasattr(order, 'order_configuration'):
                    logger.info(f"  Order Configuration: {order.order_configuration}")
                if hasattr(order, 'order_id'):
                    logger.info(f"  Order ID: {order.order_id}")
            else:
                logger.info(f"  Response is None")
            
            # Handle the response - check if order was successful
            success = False
            order_id = None
            error_details = None
            
            # Check for success/failure in response
            if hasattr(order, 'success'):
                success = order.success
            elif isinstance(order, dict):
                success = order.get('success', True)  # Default to True if not specified
            
            # Extract order ID if available
            if hasattr(order, 'order_id'):
                order_id = order.order_id
            elif hasattr(order, 'id'):
                order_id = order.id
            elif isinstance(order, dict):
                order_id = order.get('order_id') or order.get('id')
            
            # Extract error details if failed
            if not success:
                if hasattr(order, 'error_response'):
                    error_details = order.error_response
                elif isinstance(order, dict):
                    error_details = order.get('error_response')
            
            if success and order_id:
                logger.info(f"✅ Order placed successfully: {order_id}")
                return True
            elif success:
                logger.info(f"✅ Order placed successfully (response: {order})")
                return True
            else:
                logger.error(f"❌ Order failed: {error_details}")
                logger.info(f"Full response: {order}")
                return False
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            logger.debug(f"Full error details: {type(e).__name__}: {str(e)}")
            return False

class CoinbaseSmartAllocationBot:
    def __init__(self, portfolio_name: str = "coinbase_majors", strategy: str = "moderate"):
        self.portfolio_name = portfolio_name
        self.strategy = strategy
        self.data_fetcher = CoinbaseDataFetcher()
        self.trading_engine = CoinbaseTradingEngine()
        self.llm_client = None  # Will be initialized in async context
        
        # Get portfolio configuration
        if portfolio_name not in ASSET_PORTFOLIOS:
            raise ValueError(f"Portfolio '{portfolio_name}' not found in asset_config.py")
        
        self.portfolio = ASSET_PORTFOLIOS[portfolio_name]
        self.assets = [asset['symbol'] for asset in self.portfolio]
        
        # Strategy parameters
        self.min_capital = 1  # Minimum capital to start trading
        self.min_liquidation_value = 1.0  # Minimum position value to liquidate (€)
        
        logger.info("Initialized Coinbase Smart Allocation Bot (Advanced API)")
        logger.info(f"Portfolio: {portfolio_name} ({len(self.portfolio)} assets)")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"LLM-Driven Asset Selection: All available assets")
        logger.info(f"Historical Analysis: ~11-month average (330 days)")
        logger.info(f"Min Historical Data: 250+ days required")
        logger.info(f"Min Capital: €{self.min_capital}")
        logger.info(f"Min Liquidation Value: €{self.min_liquidation_value}")
        logger.info(f"Trading: {'PAPER' if not config.trading_enabled else 'LIVE'}")

    def load_position_history(self) -> Dict[str, Dict[str, float]]:
        """Load position history from file to track original buy prices"""
        try:
            import json
            history_file = "position_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load position history: {e}")
        return {}
    
    def save_position_history(self, positions: Dict[str, Dict[str, float]]):
        """Save position history to file"""
        try:
            import json
            history_file = "position_history.json"
            with open(history_file, 'w') as f:
                json.dump(positions, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save position history: {e}")
    
    def cleanup_position_history(self):
        """Clean up position_history.json by removing entries for positions that no longer exist in Coinbase"""
        try:
            logger.info("🧹 Cleaning up position_history.json...")
            
            # Load current position history
            position_history = self.load_position_history()
            if position_history is None or len(position_history) == 0:
                logger.info("📄 No position history to clean up")
                return
            
            # Detect actual positions from Coinbase
            actual_positions = self.detect_existing_positions()
            actual_symbols = set(actual_positions.keys())
            
            # Find positions in history that no longer exist in Coinbase
            history_symbols = set(position_history.keys())
            orphaned_symbols = history_symbols - actual_symbols
            
            if not orphaned_symbols:
                logger.info("✅ Position history is already clean - all entries have corresponding Coinbase positions")
                return
            
            # Remove orphaned entries
            original_count = len(position_history)
            for symbol in orphaned_symbols:
                removed_entry = position_history.pop(symbol, None)
                if removed_entry:
                    logger.info(f"🗑️ Removed {symbol} from position history (no longer held in Coinbase)")
            
            # Save cleaned position history
            self.save_position_history(position_history)
            
            new_count = len(position_history)
            logger.info(f"✅ Cleanup complete: Removed {original_count - new_count} orphaned entries")
            logger.info(f"📊 Position history now contains {new_count} entries")
            
        except Exception as e:
            logger.error(f"Error cleaning up position history: {e}")
    
    def record_buy_order(self, symbol: str, amount: float, price: float):
        """Record a buy order in position history"""
        position_history = self.load_position_history()
        
        if symbol not in position_history:
            position_history[symbol] = {
                'buy_price': price,
                'total_amount': amount,
                'buy_timestamp': datetime.now().isoformat()
            }
        else:
            # Average down the buy price if adding to existing position
            existing = position_history[symbol]
            existing_amount = existing.get('total_amount', 0)
            existing_price = existing.get('buy_price', price)
            
            # Calculate weighted average buy price
            total_cost = (existing_amount * existing_price) + (amount * price)
            new_total_amount = existing_amount + amount
            new_avg_price = total_cost / new_total_amount
            
            position_history[symbol] = {
                'buy_price': new_avg_price,
                'total_amount': new_total_amount,
                'buy_timestamp': existing.get('buy_timestamp', datetime.now().isoformat()),
                'last_add_timestamp': datetime.now().isoformat()
            }
        
        self.save_position_history(position_history)
        logger.info(f"📝 Recorded buy: {amount:.6f} {symbol} at €{price:.6f} (avg buy price: €{position_history[symbol]['buy_price']:.6f})")

    def get_buy_price(self, market_data: Dict[str, Any]) -> float:
        """Get the price to use when buying (ASK price)"""
        ask_price = market_data.get('ask', 0)
        current_price = market_data.get('current_price', 0)
        weekly_avg = market_data.get('weekly_average', current_price)
        
        # Sanity check: if ask price seems unreasonable compared to current price, use current price
        if ask_price > 0 and current_price > 0 and weekly_avg > 0:
            # If ask is less than 50% of weekly average, it's probably wrong
            if ask_price < (weekly_avg * 0.5):
                logger.warning(f"Ask price {ask_price} seems too low vs weekly avg {weekly_avg}. Using current price for buy.")
                return current_price
            return ask_price
        # Fallback to current price if ask not available or unreasonable
        return current_price
    
    def get_sell_price(self, market_data: Dict[str, Any]) -> float:
        """Get the price to use when selling or valuing positions (BID price with enhanced validation)"""
        bid_price = market_data.get('bid', 0)
        ask_price = market_data.get('ask', 0)
        current_price = market_data.get('current_price', 0)
        weekly_avg = market_data.get('weekly_average', current_price)
        yearly_avg = market_data.get('yearly_average', current_price)
        symbol = market_data.get('symbol', 'Unknown')
        
        # Enhanced BID price validation with multiple checks
        if bid_price > 0:
            reasons_to_reject = []
            
            # Check 1: BID vs weekly average (relaxed from 50% to 30%)
            if weekly_avg > 0 and bid_price < (weekly_avg * 0.3):
                reasons_to_reject.append(f"BID €{bid_price:.6f} < 30% of weekly avg €{weekly_avg:.6f}")
            
            # Check 2: BID vs ASK spread (shouldn't be too wide)
            if ask_price > 0:
                spread_pct = ((ask_price - bid_price) / bid_price) * 100
                if spread_pct > 50:  # More than 50% spread is suspicious
                    reasons_to_reject.append(f"BID-ASK spread {spread_pct:.1f}% too wide (BID €{bid_price:.6f}, ASK €{ask_price:.6f})")
            
            # Check 3: BID vs yearly average (very relaxed check)
            if yearly_avg > 0 and bid_price < (yearly_avg * 0.1):
                reasons_to_reject.append(f"BID €{bid_price:.6f} < 10% of yearly avg €{yearly_avg:.6f}")
            
            # If BID passes all checks, use it
            if not reasons_to_reject:
                logger.debug(f"💰 {symbol}: Using BID price €{bid_price:.6f} ✓ (passed all validation checks)")
                return bid_price
            else:
                # Log why BID was rejected
                logger.warning(f"⚠️ {symbol}: BID price rejected - {'; '.join(reasons_to_reject)}")
                logger.warning(f"   Fallback: Using mid-price €{current_price:.6f} instead of BID €{bid_price:.6f}")
                return current_price
        
        # Fallback: no BID price available
        logger.debug(f"💰 {symbol}: No BID price available, using mid-price €{current_price:.6f}")
        return current_price

    def get_eur_balance(self) -> float:
        """Get only EUR cash balance (excluding crypto holdings) with retry logic"""
        if not config.trading_enabled:
            return self.trading_engine.paper_balance
        
        # Retry logic for network issues
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                accounts = self.trading_engine.client.get_accounts()
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error getting EUR balance (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to get EUR balance after {max_retries} attempts: {e}")
                    return 0.0  # Return 0 on final failure
        
        try:
            
            for account in accounts['accounts'] if isinstance(accounts, dict) else accounts.accounts:
                # Always handle as dict for robustness
                if isinstance(account, dict):
                    currency = account.get('currency', '')
                    available_balance = account.get('available_balance', {})
                    value = available_balance.get('value', 0)
                else:
                    currency = getattr(account, 'currency', '')
                    available_balance = getattr(account, 'available_balance', {})
                    value = available_balance.get('value', 0)
                
                if currency == 'EUR':
                    try:
                        return float(value)
                    except Exception:
                        logger.warning(f"Could not convert EUR balance value: {value}")
                        return 0.0
            
            return 0.0
        except Exception as e:
            logger.error(f"Error getting EUR balance: {e}")
            return 0.0

    def detect_existing_positions(self) -> Dict[str, Dict[str, float]]:
        """Detect existing crypto positions from Coinbase account with retry logic"""
        positions = {}
        
        if not config.trading_enabled:
            # Convert paper positions to the expected format
            paper_positions = self.trading_engine.get_paper_positions()
            for symbol, amount in paper_positions.items():
                market_data = self.data_fetcher.get_market_data(symbol)
                if market_data and market_data.get('current_price', 0) > 0:
                    # Use sell price (bid) for position valuation - what we could sell for
                    current_price = self.get_sell_price(market_data)
                    eur_value = amount * current_price
                    positions[symbol] = {
                        'amount': amount,
                        'current_price': current_price,
                        'eur_value': eur_value
                    }
            return positions
        
        # Retry logic for network issues
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                accounts = self.trading_engine.client.get_accounts()
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to get accounts after {max_retries} attempts: {e}")
                    return positions  # Return empty positions on final failure
        
        try:
            for account in accounts['accounts'] if isinstance(accounts, dict) else accounts.accounts:
                # Always handle as dict for robustness
                if isinstance(account, dict):
                    currency = account.get('currency', '')
                    available_balance = account.get('available_balance', {})
                    value = available_balance.get('value', 0)
                else:
                    currency = getattr(account, 'currency', '')
                    available_balance = getattr(account, 'available_balance', {})
                    value = available_balance.get('value', 0)
                
                # Skip EUR and zero balances
                if currency == 'EUR':
                    continue
                    
                try:
                    crypto_amount = float(value)
                    if crypto_amount > 0:
                        # Get current price in EUR
                        symbol = f"{currency}/EUR"
                        market_data = self.data_fetcher.get_market_data(symbol)
                        if market_data and market_data.get('current_price', 0) > 0:
                            # Use sell price (bid) for position valuation - what we could sell for
                            current_price = self.get_sell_price(market_data)
                            eur_value = crypto_amount * current_price
                            # Get original buy price from history or add current price if missing
                            position_history = self.load_position_history()
                            buy_price = None
                            profit_loss_pct = None
                            position_history_updated = False
                            
                            if symbol in position_history:
                                buy_price = position_history[symbol].get('buy_price')
                                if buy_price:
                                    profit_loss_pct = ((current_price - buy_price) / buy_price) * 100
                                    logger.info(f"📊 Found buy price for {symbol} in position history: €{buy_price:.6f}")
                                else:
                                    logger.warning(f"⚠️ Position history exists for {symbol} but buy_price field is missing or None")
                            else:
                                logger.warning(f"⚠️ No position history found for {symbol} in position_history.json")
                            
                            # If no buy price history, use current price and add to position history
                            if buy_price is None:
                                buy_price = current_price  # Use current price as requested
                                profit_loss_pct = 0.0  # No profit/loss since we're setting current price as buy price
                                logger.info(f"📊 Adding missing position to history: {symbol} with current price €{buy_price:.6f}")
                                
                                # Add the missing position to position history
                                position_history[symbol] = {
                                    'buy_price': buy_price,
                                    'total_amount': crypto_amount,
                                    'buy_timestamp': datetime.now().isoformat(),
                                    'auto_added': True  # Flag to indicate this was auto-added
                                }
                                position_history_updated = True
                            
                            # Save updated position history if we added a missing position
                            if position_history_updated:
                                self.save_position_history(position_history)
                                logger.info(f"💾 Updated position_history.json with missing position: {symbol}")
                            
                            positions[symbol] = {
                                'amount': crypto_amount,
                                'current_price': current_price,
                                'buy_price': buy_price,
                                'eur_value': eur_value,
                                'profit_loss_pct': profit_loss_pct,
                                'profit_loss_eur': crypto_amount * (current_price - buy_price)
                            }
                            
                            profit_indicator = "📈" if profit_loss_pct > 0 else "📉" if profit_loss_pct < 0 else "➡️"
                            logger.info(f"📊 {currency}: {crypto_amount:.4f} = €{eur_value:.2f} (€{current_price:.2f}) {profit_indicator} {profit_loss_pct:+.1f}%")
                except Exception as e:
                    logger.warning(f"Could not process {currency} position: {e}")
        
        except Exception as e:
            logger.error(f"Error detecting positions: {e}")
        
        return positions
    
    def save_analysis_data(self, data: Dict[str, Any], analysis_type: str = "smart_allocation"):
        """Save analysis data to the analysis folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{analysis_type}_{self.portfolio_name}_{timestamp}.json"
        filepath = os.path.join(ANALYSIS_FOLDER, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Analysis data saved to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving analysis data: {e}")
            return None
    
    def load_historical_analysis(self, analysis_type: str = "smart_allocation", limit: int = 10):
        """Load historical analysis data from the analysis folder"""
        try:
            analysis_files = []
            for filename in os.listdir(ANALYSIS_FOLDER):
                if filename.startswith(analysis_type) and filename.endswith('.json'):
                    filepath = os.path.join(ANALYSIS_FOLDER, filename)
                    analysis_files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            analysis_files.sort(key=lambda x: x[1], reverse=True)
            
            historical_data = []
            for filepath, _ in analysis_files[:limit]:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        historical_data.append(data)
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}")
            
            return historical_data
        except Exception as e:
            logger.error(f"Error loading historical analysis: {e}")
            return []
    
    async def initialize(self):
        """Initialize the bot and test connections"""
        logger.info("Initializing Coinbase Smart Allocation Bot...")
        
        # NOTE: No connection tests during initialization to avoid accidental trades
        
        # Get account balance
        balance = self.trading_engine.get_account_balance()
        if not config.trading_enabled:
            logger.info(f"📊 Paper Trading Balance: €{balance:.2f} EUR")
        else:
            logger.info(f"💰 Account Balance: €{balance:.2f}")
        
        logger.info("Coinbase Smart Allocation Bot initialized successfully!")
    
    async def sell_specific_crypto(self, symbol: str) -> bool:
        """Sell a specific cryptocurrency on startup
        
        Args:
            symbol: The crypto symbol to sell (e.g., 'BTC', 'ETH', 'SOL')
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"🔴 STARTUP SELL: Attempting to sell all {symbol} holdings...")
        
        # Normalize symbol to include /EUR
        if '/' not in symbol:
            symbol = f"{symbol}/EUR"
        
        # Detect existing positions
        existing_positions = self.detect_existing_positions()
        
        # Find the position to sell
        position = existing_positions.get(symbol)
        if not position:
            logger.warning(f"⚠️ No {symbol} position found to sell")
            return False
        
        amount = position['amount']
        eur_value = position['eur_value']
        
        logger.info(f"📊 Found {symbol} position: {amount:.6f} tokens = €{eur_value:.2f}")
        
        # Skip if position is too small
        if eur_value < self.min_liquidation_value:
            logger.warning(f"⚠️ {symbol} position too small (€{eur_value:.2f} < €{self.min_liquidation_value:.2f}). Skipping.")
            return False
        
        # Execute the sell order
        logger.info(f"🔴 Selling {amount:.6f} {symbol} at market price...")
        success = self.trading_engine.place_order(symbol, 'sell', amount, None)  # Market order
        
        if success:
            logger.info(f"✅ Successfully sold {amount:.6f} {symbol} (€{eur_value:.2f})")
            return True
        else:
            logger.error(f"❌ Failed to sell {symbol}")
            return False

    async def run_smart_allocation_cycle(self):
        """Run one complete smart allocation cycle with LLM-confirmed buy and sell logic and full LLM request/response logging"""
        logger.info("🔄 Starting smart allocation cycle...")
        
        # Clean up position history (remove entries for positions no longer held in Coinbase)
        self.cleanup_position_history()
        
        # Detect existing positions from Coinbase account
        existing_positions = self.detect_existing_positions()
        
        # Get current portfolio value (total EUR value including crypto)
        portfolio_value = self.trading_engine.get_account_balance()
        # Get available EUR cash (excluding crypto holdings)
        available_capital = self.get_eur_balance()
        
        logger.info(f"📊 Total Portfolio Value: €{portfolio_value:.2f}")
        logger.info(f"💰 Available EUR Cash: €{available_capital:.2f}")
        
        # Log existing positions
        if existing_positions:
            logger.info("🏦 Existing positions detected:")
            for symbol, position in existing_positions.items():
                logger.info(f"  {symbol.split('/')[0]}: {position['amount']:.4f} = €{position['eur_value']:.2f} (€{position['current_price']:.2f})")
        else:
            logger.info("🏦 No existing positions detected")
        
        async with LLMClient() as llm_client:
            # 1. INDIVIDUAL POSITION MANAGEMENT: Each holding gets its own isolated LLM decision
            if existing_positions:
                logger.info(f"🔍 Evaluating {len(existing_positions)} positions individually...")
                for symbol, position in existing_positions.items():
                    amount = position['amount']
                    current_price = position['current_price']
                    eur_value = position['eur_value']
                    
                    logger.info(f"🎯 ISOLATED DECISION #{list(existing_positions.keys()).index(symbol) + 1}: {symbol.split('/')[0]} {amount:.4f} = €{eur_value:.2f} (€{current_price:.2f})")
                    
                    # Get full market data for this asset to send to LLM
                    market_data = self.data_fetcher.get_market_data(symbol)
                    if market_data:
                        # Prepare comprehensive market data for LLM with ALL available information
                        yearly_avg = market_data.get('yearly_average', current_price)
                        weekly_avg = market_data.get('weekly_average', current_price)
                        
                        # Calculate price vs both averages - let LLM decide what's relevant
                        if yearly_avg > 0:
                            price_vs_yearly_avg_pct = ((current_price - yearly_avg) / yearly_avg) * 100
                        else:
                            price_vs_yearly_avg_pct = 0
                            
                        if weekly_avg > 0:
                            price_vs_weekly_avg_pct = ((current_price - weekly_avg) / weekly_avg) * 100
                        else:
                            price_vs_weekly_avg_pct = 0
                        
                        # Use sell price (bid) for current valuation
                        sell_price = self.get_sell_price(market_data)
                        buy_price_for_reference = self.get_buy_price(market_data)
                        
                        # Enhanced price data for LLM with full transparency
                        raw_bid = float(market_data.get('bid', 0))
                        raw_ask = float(market_data.get('ask', 0))
                        mid_price = (raw_bid + raw_ask) / 2 if raw_bid > 0 and raw_ask > 0 else current_price
                        spread_pct = ((raw_ask - raw_bid) / mid_price * 100) if raw_bid > 0 and raw_ask > 0 else 0
                        
                        llm_sell_request = {
                            'symbol': symbol,
                            'current_price': float(sell_price),  # SELL/VALUATION price (BID or fallback)
                            'buy_price': float(position.get('buy_price', current_price)),
                            'profit_loss_pct': float(position.get('profit_loss_pct', 0)),
                            'profit_loss_eur': float(position.get('profit_loss_eur', 0)),
                            'price_vs_yearly_avg_pct': float(price_vs_yearly_avg_pct),  # Provide yearly comparison
                            'price_vs_weekly_avg_pct': float(price_vs_weekly_avg_pct),  # Provide weekly comparison
                            'volume_24h': market_data.get('volume', 0),
                            'yearly_average': float(yearly_avg),  # Provide yearly average
                            'weekly_average': float(weekly_avg),  # Provide weekly average
                            # ENHANCED PRICE TRANSPARENCY FOR LLM
                            'latest_bid': float(raw_bid),  # Exact latest BID from Coinbase
                            'latest_ask': float(raw_ask),  # Exact latest ASK from Coinbase
                            'mid_price': float(mid_price),  # Mid-point between BID/ASK
                            'spread_percent': float(spread_pct),  # BID-ASK spread as percentage
                            'valuation_price': float(sell_price),  # Price used for position valuation
                            'buy_order_price': float(buy_price_for_reference),  # Price that would be used for buying
                            'price_explanation': f"Valuation uses BID (€{raw_bid:.6f}) {'✓' if sell_price == raw_bid else '✗ fallback to current'}, Buy would use ASK (€{raw_ask:.6f})",
                            # LEGACY FIELDS (for backward compatibility)
                            'bid': float(raw_bid),
                            'ask': float(raw_ask),
                            # Technical indicators
                            'rsi': float(market_data.get('rsi', 50)),
                            'macd': float(market_data.get('macd', 0)),
                            'ma_20': float(market_data.get('ma_20', current_price)),
                            'ma_50': float(market_data.get('ma_50', current_price)),
                            'current_position': f"{amount:.6f} {symbol.split('/')[0]}",
                            'portfolio_value': float(portfolio_value),
                            'position_value_eur': float(eur_value),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        # Fallback to basic data if market data unavailable - provide all data points
                        llm_sell_request = {
                            'symbol': symbol,
                            'current_price': float(current_price),
                            'price_vs_yearly_avg_pct': 0,  # No historical data available
                            'price_vs_weekly_avg_pct': 0,  # No historical data available
                            'volume_24h': 0,
                            'yearly_average': float(current_price),  # Fallback to current price
                            'weekly_average': float(current_price),  # Fallback to current price
                            'rsi': 50,  # Neutral RSI
                            'macd': 0,  # Neutral MACD
                            'ma_20': float(current_price),
                            'ma_50': float(current_price),
                            'current_position': f"{amount:.6f} {symbol.split('/')[0]}",
                            'portfolio_value': float(portfolio_value),
                            'position_value_eur': float(eur_value),
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    logger.info(f"🤖 LLM sell request for {symbol} (ISOLATED):")
                    logger.info(f"  Sell Price (Bid): €{llm_sell_request.get('current_price', 0):.2f}")
                    logger.info(f"  Original Buy Price: €{llm_sell_request.get('buy_price', 0):.2f}")
                    logger.info(f"  Profit/Loss: {llm_sell_request.get('profit_loss_pct', 0):+.1f}% (€{llm_sell_request.get('profit_loss_eur', 0):+.2f})")
                    logger.info(f"  Position: {llm_sell_request.get('current_position', 'Unknown')}")
                    logger.info(f"📊 DATA TO LLM: Providing comprehensive market data (yearly avg: €{llm_sell_request.get('yearly_average', 0):.6f}, weekly avg: €{llm_sell_request.get('weekly_average', 0):.6f}) for LLM decision")
                    
                    llm_sell_decision = await llm_client.get_trading_decision(llm_sell_request)
                    # Note: LLM client now logs its own formatted decision, so we don't need to log it again here
                    
                    if isinstance(llm_sell_decision, dict) and llm_sell_decision.get('action', '').upper() == 'SELL':
                        logger.info(f"🔴 LLM DECIDED: SELL {amount:.6f} {symbol}. Bot executing LLM's decision...")
                        # Use market order to sell all holdings
                        success = self.trading_engine.place_order(symbol, 'sell', amount, None)  # Market order
                        if success:
                            logger.info(f"✅ EXECUTED: Sold {amount:.6f} {symbol} at market price per LLM decision.")
                            # Update available capital after sale
                            available_capital = self.get_eur_balance()
                            logger.info(f"💰 Updated EUR balance: €{available_capital:.2f}")
                        else:
                            logger.warning(f"❌ Failed to execute LLM's sell decision for {symbol}. Will retry next cycle.")
                            # Continue to next position instead of returning
                            continue
                    else:
                        logger.info(f"🟢 LLM DECIDED: HOLD {symbol}. Bot taking no action per LLM decision.")
            
            # 2. SEPARATE EUR INVESTMENT DECISION: If have EUR cash, use LLM to select best asset
            logger.info("💰 ISOLATED EUR INVESTMENT DECISION: Evaluating available EUR cash...")
            
            # NEW LOGIC: Only invest if available cash > €10, then invest full amount
            if available_capital > 10.0:
                # Reserve only €1 for fees/slippage, invest the rest
                fees_reserve = 1.0
                investable_capital = available_capital - fees_reserve
                
                logger.info(f"🎯 EUR INVESTMENT: €{available_capital:.2f} available, investing €{investable_capital:.2f} (fees reserve: €{fees_reserve:.2f})")
                all_market_data = [self.data_fetcher.get_market_data(asset) for asset in self.assets]
                all_market_data = [d for d in all_market_data if d]
                if not all_market_data:
                    logger.warning("No market data available to allocate capital.")
                    return
                
                # Add timestamp to each asset
                for asset_data in all_market_data:
                    asset_data['timestamp'] = datetime.now().isoformat()
                
                logger.info(f"🤖 LLM asset selection request (ISOLATED):")
                logger.info(f"  Available Assets: {len(all_market_data)}")
                logger.info(f"  Total Capital: €{available_capital:.2f}")
                logger.info(f"  Fees Reserve: €{fees_reserve:.2f}")
                logger.info(f"  Investable: €{investable_capital:.2f}")
                
                best_asset = await llm_client.get_asset_selection(all_market_data, portfolio_value, investable_capital)
                # Note: LLM client now logs its own formatted response
                
                if isinstance(best_asset, dict) and 'symbol' in best_asset:
                    symbol = best_asset['symbol']
                else:
                    symbol = all_market_data[0]['symbol']
                    
                selected = next((d for d in all_market_data if d['symbol'] == symbol), all_market_data[0])
                logger.info(f"LLM selected {symbol} for BUY allocation: €{selected['current_price']:.6f} (yearly avg: €{selected['yearly_average']:.6f}, weekly avg: €{selected['weekly_average']:.6f})")
                logger.info(f"📊 LLM SELECTED: {symbol} - Bot will execute LLM's buy decision")
                
                # Calculate quantity to buy - use ask price for buy orders
                buy_price = self.get_buy_price(selected)  # Use ask price for buying
                
                # First format the price to see what Coinbase will actually accept
                product_id = symbol.replace('/', '-')
                _, formatted_price_str = self.trading_engine.format_order_params(product_id, 1.0, buy_price)
                formatted_price = float(formatted_price_str) if formatted_price_str else buy_price
                
                # For market orders, we specify the EUR amount to spend (quote_size)
                # Invest the full investable capital (all available cash minus €1 fees reserve)
                quote_amount = round(investable_capital, 2)  # Round to 2 decimal places for EUR precision
                logger.info(f"Market buy order: spending €{quote_amount:.2f} EUR at ask price €{buy_price:.6f} (full available capital minus €{fees_reserve:.2f} fees reserve)")
                success = self.trading_engine.place_order(symbol, 'buy', quote_amount, None)  # Market order with EUR amount
                if success:
                    # Calculate approximate amount purchased (we don't know exact amount until order executes)
                    approx_crypto_amount = quote_amount / buy_price
                    # Record the buy order for position tracking using buy price
                    self.record_buy_order(symbol, approx_crypto_amount, buy_price)
                    logger.info(f"✅ EXECUTED: Market buy per LLM decision - spent €{quote_amount:.2f} EUR on {symbol} (approx {approx_crypto_amount:.6f} tokens).")
                else:
                    logger.warning(f"❌ Failed to execute LLM's buy decision for {symbol}. Will retry next cycle.")
            else:
                logger.info(f"💸 Waiting for more capital: €{available_capital:.2f} available (need >€10.00 to invest)")
        
        # Save analysis data
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_name': self.portfolio_name,
            'strategy': self.strategy,
            'portfolio_value': portfolio_value,
            'available_capital': available_capital,
            'existing_positions': existing_positions,
            'trading_enabled': config.trading_enabled
        }
        self.save_analysis_data(analysis_data, "smart_allocation")
        logger.info("🔄 Smart allocation cycle complete.")
        logger.info("📋 ISOLATION SUMMARY: Each position evaluated independently, EUR investment decided separately")

    async def run_rebalancing_cycle(self) -> bool:
        """Run complete portfolio rebalancing: liquidate all holdings and reinvest in LLM's top 5 picks
        Returns True if rebalancing should allow continued operation, False if bot should stop"""
        logger.info("🔄 Starting COMPLETE PORTFOLIO REBALANCING...")
        
        # 1. LIQUIDATION PHASE: Sell all existing holdings
        logger.info("💰 PHASE 1: LIQUIDATING ALL HOLDINGS")
        existing_positions = self.detect_existing_positions()
        
        total_liquidated = 0.0
        liquidation_successes = 0
        liquidation_attempts = 0
        
        if existing_positions:
            logger.info(f"🏦 Found {len(existing_positions)} existing positions to liquidate:")
            for symbol, position in existing_positions.items():
                amount = position['amount']
                current_price = position['current_price']
                eur_value = position['eur_value']
                
                # Skip positions worth less than minimum threshold to avoid fees and failed orders
                if eur_value < self.min_liquidation_value:
                    logger.info(f"  ⏭️ Skipping {symbol.split('/')[0]}: {amount:.4f} = €{eur_value:.2f} (below €{self.min_liquidation_value} minimum)")
                    continue
                
                logger.info(f"  💸 Liquidating {symbol.split('/')[0]}: {amount:.4f} = €{eur_value:.2f}")
                
                liquidation_attempts += 1
                # Sell the position using market order
                success = self.trading_engine.place_order(symbol, 'sell', amount, None)
                if success:
                    liquidation_successes += 1
                    total_liquidated += eur_value
                    logger.info(f"✅ SOLD: {amount:.4f} {symbol} (€{eur_value:.2f})")
                else:
                    logger.warning(f"❌ Failed to sell {symbol}")
        else:
            logger.info("🏦 No existing positions to liquidate")
        
        # Get updated balance after liquidation (with retries for settlement)
        logger.info("⏰ Waiting for liquidation settlement...")
        
        # Wait a few seconds for liquidation to settle
        await asyncio.sleep(3)
        
        # Get initial balance for comparison
        initial_balance = self.get_eur_balance() if liquidation_successes == 0 else 0
        
        # Try multiple times to get the updated balance
        max_retries = 5
        for attempt in range(max_retries):
            total_capital = self.get_eur_balance()
            logger.info(f"💰 Balance check {attempt + 1}/{max_retries}: €{total_capital:.2f}")
            
            # Calculate expected minimum balance (initial balance + 90% of liquidated amount)
            expected_min_balance = initial_balance + (total_liquidated * 0.9)
            
            # If balance looks reasonable, break
            if liquidation_successes > 0 and total_capital >= expected_min_balance:
                logger.info(f"✅ Balance settlement confirmed: €{total_capital:.2f} >= €{expected_min_balance:.2f}")
                break
            elif liquidation_successes == 0:  # No liquidations, current balance is fine
                break
            
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                logger.info(f"⏰ Balance seems low (€{total_capital:.2f} < €{expected_min_balance:.2f}), waiting 2 more seconds for settlement...")
                await asyncio.sleep(2)
        
        logger.info(f"💰 Final capital after liquidation: €{total_capital:.2f}")
        
        # Summary of liquidation phase
        if liquidation_attempts > 0:
            liquidation_success_rate = (liquidation_successes / liquidation_attempts) * 100
            logger.info(f"📊 Liquidation Summary: {liquidation_successes}/{liquidation_attempts} successful ({liquidation_success_rate:.1f}%)")
            
            # If we had positions to liquidate but none succeeded, that's a critical failure
            if liquidation_successes == 0:
                logger.error("❌ CRITICAL: All liquidation attempts failed! Stopping bot.")
                return False
        elif existing_positions:
            # We had positions but skipped them all due to low value
            skipped_count = len(existing_positions)
            logger.info(f"📊 Liquidation Summary: {skipped_count} positions skipped (all below €{self.min_liquidation_value} minimum)")
            logger.info("✅ No liquidation needed - all positions below minimum threshold")
        
        # Calculate buffer - keep minimal cash buffer for rebalancing
        buffer_amount = max(2.0, min(5.0, total_capital * 0.02))  # 2% buffer, min €2, max €5
        investable_capital = total_capital - buffer_amount
        
        if investable_capital < 5.0:
            logger.warning(f"💸 Insufficient capital for reinvestment: €{investable_capital:.2f} (need ≥€5)")
            logger.info("✅ Liquidation phase completed, but skipping reinvestment due to insufficient capital")
            logger.info("🔄 REBALANCING PHASE COMPLETED (liquidation only)")
            return True  # Allow continued operation with liquidation-only rebalancing
        
        logger.info(f"🎯 Investable capital: €{investable_capital:.2f} (buffer: €{buffer_amount:.2f})")
        
        # 2. ANALYSIS PHASE: Get all market data and LLM recommendations
        logger.info("📊 PHASE 2: LLM PORTFOLIO ANALYSIS")
        
        # Gather market data for all available assets
        logger.info(f"📈 Analyzing {len(self.assets)} available cryptocurrencies...")
        all_market_data = []
        for asset in self.assets:
            market_data = self.data_fetcher.get_market_data(asset)
            if market_data:
                market_data['timestamp'] = datetime.now().isoformat()
                all_market_data.append(market_data)
        
        if len(all_market_data) < 5:
            logger.error(f"❌ Insufficient market data: only {len(all_market_data)} assets available (need ≥5)")
            return False  # Critical failure - stop bot operation
        
        logger.info(f"✅ Market data gathered for {len(all_market_data)} cryptocurrencies")
        
        async with LLMClient() as llm_client:
            # Get LLM's top 5 crypto recommendations
            logger.info("🤖 REQUESTING TOP 5 CRYPTO RECOMMENDATIONS...")
            top_cryptos = await llm_client.get_top_crypto_recommendations(all_market_data, investable_capital)
            
            recommended_symbols = top_cryptos.get('recommended_cryptos', [])
            if len(recommended_symbols) != 5:
                logger.error(f"❌ LLM returned {len(recommended_symbols)} recommendations (expected 5)")
                return False  # Critical failure - stop bot operation
            
            logger.info("🎯 LLM SELECTED TOP 5 CRYPTOCURRENCIES:")
            for i, symbol in enumerate(recommended_symbols, 1):
                logger.info(f"  {i}. {symbol}")
            
            # Get LLM's allocation strategy
            logger.info("🤖 REQUESTING PORTFOLIO ALLOCATION STRATEGY...")
            allocation_decision = await llm_client.get_portfolio_allocation(
                recommended_symbols, investable_capital, all_market_data
            )
            
            allocations = allocation_decision.get('allocations', {})
            if not allocations:
                logger.error("❌ LLM failed to provide allocations. Using equal weights.")
                allocations = {symbol: 20.0 for symbol in recommended_symbols}  # Equal 20% each
            
            # 3. INVESTMENT PHASE: Execute the new portfolio
            logger.info("💰 PHASE 3: EXECUTING NEW PORTFOLIO")
            
            successful_investments = 0
            total_invested = 0.0
            
            for symbol, percentage in allocations.items():
                eur_amount = (percentage / 100) * investable_capital
                
                logger.info(f"🎯 Investing {percentage:.1f}% (€{eur_amount:.2f}) in {symbol}")
                
                # Get current market data for this specific asset
                asset_data = next((d for d in all_market_data if d.get('symbol') == symbol), None)
                if not asset_data:
                    logger.warning(f"❌ No market data for {symbol}, skipping")
                    continue
                
                # Use ask price for buying
                buy_price = self.get_buy_price(asset_data)
                
                # Use 99.5% of allocated amount to account for fees and slippage 
                safe_eur_amount = round(eur_amount * 0.995, 2)
                
                # Execute market buy order
                success = self.trading_engine.place_order(symbol, 'buy', safe_eur_amount, None)
                if success:
                    # Calculate approximate amount purchased
                    approx_crypto_amount = safe_eur_amount / buy_price
                    # Record the buy order for position tracking
                    self.record_buy_order(symbol, approx_crypto_amount, buy_price)
                    
                    successful_investments += 1
                    total_invested += safe_eur_amount
                    logger.info(f"✅ INVESTED: €{safe_eur_amount:.2f} in {symbol} (≈{approx_crypto_amount:.4f} tokens)")
                else:
                    logger.warning(f"❌ Failed to invest in {symbol}")
            
            # 4. SUMMARY PHASE
            logger.info("📊 REBALANCING COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"💸 Total Liquidated: €{total_liquidated:.2f}")
            logger.info(f"💰 Total Capital: €{total_capital:.2f}")
            logger.info(f"🎯 Successfully Invested: €{total_invested:.2f} in {successful_investments}/5 assets")
            logger.info(f"💵 Remaining Cash: €{total_capital - total_invested:.2f}")
            logger.info("📈 NEW PORTFOLIO:")
            
            for symbol, percentage in allocations.items():
                eur_amount = (percentage / 100) * investable_capital
                logger.info(f"  {symbol}: {percentage:.1f}% (€{eur_amount:.2f})")
            
            # Save rebalancing analysis
            rebalancing_data = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_name': self.portfolio_name,
                'strategy': self.strategy,
                'liquidated_positions': existing_positions,
                'total_liquidated': total_liquidated,
                'total_capital': total_capital,
                'investable_capital': investable_capital,
                'recommended_cryptos': recommended_symbols,
                'allocations': allocations,
                'successful_investments': successful_investments,
                'total_invested': total_invested,
                'llm_reasoning': {
                    'crypto_selection': top_cryptos.get('reasoning', ''),
                    'allocation_strategy': allocation_decision.get('reasoning', '')
                }
            }
            self.save_analysis_data(rebalancing_data, "rebalancing")
            logger.info("💾 Rebalancing analysis saved")
            logger.info("🔄 PORTFOLIO REBALANCING COMPLETE!")
            
            return True  # Successful rebalancing - continue with normal operation

async def main():
    """Main function to run the bot"""
    bot = CoinbaseSmartAllocationBot()
    await bot.run_smart_allocation_cycle()

if __name__ == "__main__":
    asyncio.run(main()) 