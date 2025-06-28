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
            
            # Get historical candles for yearly average (limited to 330 days due to Coinbase 350 candle limit)
            now = datetime.now(timezone.utc)
            start_dt = now - timedelta(days=330)  # Reduced from 365 to stay under 350 candle limit
            start = int(start_dt.timestamp())
            end = int(now.timestamp())
            
            # Try to get daily candles for ~11 months (more efficient than hourly)
            try:
                candles = self.client.get_candles(product_id=product_id, start=start, end=end, granularity='ONE_DAY')
                closes = [float(candle.close) for candle in candles.candles]
                
                # Validate we have at least 250 days of data (allowing for some missing days and weekends)
                if len(closes) < 250:
                    logger.warning(f"{symbol}: Only {len(closes)} days of historical data available (need 250+). Skipping this asset.")
                    return None  # Skip assets without sufficient historical data
                
                yearly_avg = np.mean(closes) if closes else 0
                logger.info(f"{symbol}: Using {len(closes)} days of historical data for ~11-month average")
                
            except Exception as candle_error:
                logger.warning(f"Failed to get daily candles for {symbol}, trying hourly: {candle_error}")
                # Fallback to hourly candles for last 30 days if daily fails
                start_dt = now - timedelta(days=30)
                start = int(start_dt.timestamp())
                candles = self.client.get_candles(product_id=product_id, start=start, end=end, granularity='ONE_HOUR')
                closes = [float(candle.close) for candle in candles.candles]
                
                if len(closes) < 600:  # Less than ~25 days of hourly data
                    logger.warning(f"{symbol}: Insufficient historical data even with hourly fallback. Skipping this asset.")
                    return None
                
                yearly_avg = np.mean(closes) if closes else 0
                logger.warning(f"{symbol}: Using fallback 30-day average instead of ~11-month average (insufficient data)")
            
            # If current price seems wrong (too low compared to yearly average), use most recent candle close
            if yearly_avg > 0 and current_price > 0 and current_price < (yearly_avg * 0.1):  # Current price is less than 10% of yearly average
                logger.warning(f"Current price {current_price} seems too low vs yearly avg {yearly_avg}. Using latest candle close.")
                if closes:
                    corrected_price = float(closes[-1])  # Convert to regular Python float
                    # Sanity check: corrected price should be within reasonable range of yearly average
                    if corrected_price > 0 and corrected_price < (yearly_avg * 10):  # Less than 10x yearly average
                        current_price = corrected_price
                        logger.info(f"Updated current price to: {current_price}")
                    else:
                        logger.warning(f"Corrected price {corrected_price} seems unreasonable vs yearly avg {yearly_avg}. Using yearly average.")
                        current_price = float(yearly_avg)
                else:
                    logger.warning("No candle data available. Using yearly average.")
                    current_price = float(yearly_avg)
            
            # Calculate technical indicators from historical data
            indicators = self._calculate_technical_indicators(closes) if closes else {}
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'yearly_average': yearly_avg,
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
                    if not ticker or not ticker.get('bids') or not ticker.get('asks'):
                        continue
                        
                    bid_price = float(ticker['bids'][0]['price']) if ticker.get('bids') else None
                    ask_price = float(ticker['asks'][0]['price']) if ticker.get('asks') else None
                    current_price = (bid_price + ask_price) / 2 if bid_price and ask_price else None
                    
                    if not current_price:
                        continue
                    
                    # Get historical data for yearly average and technical indicators
                    try:
                        # Try ~11-month data first (330 days to stay under 350 candle limit)
                        candles = self.client.get_product_candles(
                            product_id=product_id,
                            start=int((datetime.now() - timedelta(days=330)).timestamp()),
                            end=int(datetime.now().timestamp()),
                            granularity='ONE_DAY'
                        )
                        
                        if candles and candles.get('candles') and len(candles['candles']) >= 250:
                            prices = [float(candle['close']) for candle in candles['candles']]
                            yearly_average = sum(prices) / len(prices) if prices else current_price
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
                                indicators = self._calculate_technical_indicators(prices)
                            else:
                                # Skip asset if insufficient data
                                continue
                    except Exception:
                        yearly_average = current_price
                        indicators = {}
                    
                    # Success! Return the data
                    return {
                        'symbol': symbol,
                        'current_price': current_price,
                        'bid': bid_price,
                        'ask': ask_price,
                        'yearly_average': yearly_average,
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
            'SOL-EUR': {'size_decimals': 2, 'price_decimals': 2},  # SOL needs only 2 decimal places for size
            'ADA-EUR': {'size_decimals': 2, 'price_decimals': 4},
            # Default fallback
            'default': {'size_decimals': 2, 'price_decimals': 2}
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
        
        # Format amount (base_size) with proper decimal places
        formatted_amount = round(amount, rules['size_decimals'])
        formatted_amount_str = f"{formatted_amount:.{rules['size_decimals']}f}".rstrip('0').rstrip('.')
        
        # Format price with proper decimal places if provided
        formatted_price_str = None
        if price is not None:
            formatted_price = round(price, rules['price_decimals'])
            formatted_price_str = f"{formatted_price:.{rules['price_decimals']}f}".rstrip('0').rstrip('.')
        
        logger.info(f"Formatted order params for {product_id}: amount={formatted_amount_str}, price={formatted_price_str}")
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
            
            # Place order using Coinbase Advanced API
            if price:
                # Limit order
                order = self.client.create_order(
                    product_id=product_id,
                    side=side.upper(),
                    client_order_id=client_order_id,
                    order_configuration={
                        'limit_limit_gtc': {
                            'base_size': formatted_amount,
                            'limit_price': formatted_price
                        }
                    }
                )
            else:
                # Market order - use quote_size for market orders
                order = self.client.create_order(
                    product_id=product_id,
                    side=side.upper(),
                    client_order_id=client_order_id,
                    order_configuration={
                        'market_market_ioc': {
                            'quote_size': formatted_amount
                        }
                    }
                )
            
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
        
        logger.info("Initialized Coinbase Smart Allocation Bot (Advanced API)")
        logger.info(f"Portfolio: {portfolio_name} ({len(self.portfolio)} assets)")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"LLM-Driven Asset Selection: All available assets")
        logger.info(f"Historical Analysis: ~11-month average (330 days)")
        logger.info(f"Min Historical Data: 250+ days required")
        logger.info(f"Min Capital: €{self.min_capital}")
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
        """Get the price to use when selling or valuing positions (BID price)"""
        bid_price = market_data.get('bid', 0)
        current_price = market_data.get('current_price', 0)
        weekly_avg = market_data.get('weekly_average', current_price)
        
        # Sanity check: if bid price seems unreasonable compared to current price, use current price
        if bid_price > 0 and current_price > 0 and weekly_avg > 0:
            # If bid is less than 50% of weekly average, it's probably wrong
            if bid_price < (weekly_avg * 0.5):
                logger.warning(f"Bid price {bid_price} seems too low vs weekly avg {weekly_avg}. Using current price for sell.")
                return current_price
            return bid_price
        # Fallback to current price if bid not available or unreasonable
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
                            # Get original buy price from history or estimate from weekly average
                            position_history = self.load_position_history()
                            buy_price = None
                            profit_loss_pct = None
                            
                            if symbol in position_history:
                                buy_price = position_history[symbol].get('buy_price')
                                if buy_price:
                                    profit_loss_pct = ((current_price - buy_price) / buy_price) * 100
                                    logger.info(f"📊 Found buy price for {symbol} in position history: €{buy_price:.6f}")
                                else:
                                    logger.warning(f"⚠️ Position history exists for {symbol} but buy_price field is missing or None")
                            else:
                                logger.warning(f"⚠️ No position history found for {symbol} in position_history.json")
                            
                            # If no buy price history, estimate from weekly average (conservative approach)
                            if buy_price is None:
                                weekly_avg = market_data.get('weekly_average', current_price)
                                buy_price = weekly_avg  # Assume bought at weekly average
                                profit_loss_pct = ((current_price - buy_price) / buy_price) * 100
                                logger.info(f"📊 Estimating buy price for {symbol} as weekly average: €{buy_price:.6f}")
                            
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
    

    
    async def run_smart_allocation_cycle(self):
        """Run one complete smart allocation cycle with LLM-confirmed buy and sell logic and full LLM request/response logging"""
        logger.info("🔄 Starting smart allocation cycle...")
        
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
                        # Calculate price vs yearly average percentage
                        yearly_avg = market_data.get('yearly_average', current_price)
                        if yearly_avg > 0:
                            price_vs_yearly_avg_pct = ((current_price - yearly_avg) / yearly_avg) * 100
                        else:
                            price_vs_yearly_avg_pct = 0
                        
                        # Prepare comprehensive market data for LLM (convert all to regular Python types)
                        # Use sell price (bid) for current valuation
                        sell_price = self.get_sell_price(market_data)
                        llm_sell_request = {
                            'symbol': symbol,
                            'current_price': float(sell_price),  # Use sell price for valuation
                            'buy_price': float(position.get('buy_price', current_price)),
                            'profit_loss_pct': float(position.get('profit_loss_pct', 0)),
                            'profit_loss_eur': float(position.get('profit_loss_eur', 0)),
                            'price_vs_yearly_avg_pct': float(price_vs_yearly_avg_pct),
                            'volume_24h': market_data.get('volume', 0),
                            'yearly_average': float(yearly_avg),
                            'bid': float(market_data.get('bid', current_price)),
                            'ask': float(market_data.get('ask', current_price)),
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
                        # Fallback to basic data if market data unavailable
                        llm_sell_request = {
                            'symbol': symbol,
                            'current_price': float(current_price),
                            'price_vs_yearly_avg_pct': 0,
                            'volume_24h': 0,
                            'yearly_average': float(current_price),
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
                    
                    llm_sell_decision = await llm_client.get_trading_decision(llm_sell_request)
                    # Note: LLM client now logs its own formatted decision, so we don't need to log it again here
                    
                    if isinstance(llm_sell_decision, dict) and llm_sell_decision.get('action', '').upper() == 'SELL':
                        logger.info(f"🔴 SELL DECISION: LLM recommends selling {amount:.6f} {symbol}. Executing sell order...")
                        # Use market order to sell all holdings
                        success = self.trading_engine.place_order(symbol, 'sell', amount, None)  # Market order
                        if success:
                            logger.info(f"✅ SOLD: {amount:.6f} {symbol} at market price. Position closed.")
                            # Update available capital after sale
                            available_capital = self.get_eur_balance()
                            logger.info(f"💰 Updated EUR balance: €{available_capital:.2f}")
                        else:
                            logger.warning(f"❌ Failed to sell {symbol}. Will retry next cycle.")
                            # Continue to next position instead of returning
                            continue
                    else:
                        logger.info(f"🟢 HOLD DECISION: LLM recommends holding {symbol}. No action taken.")
            
            # 2. SEPARATE EUR INVESTMENT DECISION: If have EUR cash, use LLM to select best asset
            logger.info("💰 ISOLATED EUR INVESTMENT DECISION: Evaluating available EUR cash...")
            # Keep a buffer of €2-5 EUR (scale with portfolio size)
            buffer_amount = max(2.0, min(5.0, available_capital * 0.1))  # 10% buffer, min €2, max €5
            investable_capital = available_capital - buffer_amount
            
            if investable_capital >= 1:
                logger.info(f"🎯 EUR INVESTMENT: €{investable_capital:.2f} available (buffer: €{buffer_amount:.2f})")
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
                logger.info(f"  Buffer: €{buffer_amount:.2f}")
                logger.info(f"  Investable: €{investable_capital:.2f}")
                
                best_asset = await llm_client.get_asset_selection(all_market_data, portfolio_value, investable_capital)
                # Note: LLM client now logs its own formatted response
                
                if isinstance(best_asset, dict) and 'symbol' in best_asset:
                    symbol = best_asset['symbol']
                else:
                    symbol = all_market_data[0]['symbol']
                    
                selected = next((d for d in all_market_data if d['symbol'] == symbol), all_market_data[0])
                logger.info(f"LLM selected {symbol} for allocation: €{selected['current_price']:.6f} (yearly avg: €{selected['yearly_average']:.6f})")
                
                # Calculate quantity to buy - use ask price for buy orders
                buy_price = self.get_buy_price(selected)  # Use ask price for buying
                
                # First format the price to see what Coinbase will actually accept
                product_id = symbol.replace('/', '-')
                _, formatted_price_str = self.trading_engine.format_order_params(product_id, 1.0, buy_price)
                formatted_price = float(formatted_price_str) if formatted_price_str else buy_price
                
                # For market orders, we specify the EUR amount to spend (quote_size)
                # Use 95% of investable capital (which already excludes the buffer)
                safe_quote_amount = round(investable_capital * 0.95, 2)  # Round to 2 decimal places for EUR precision
                logger.info(f"Market buy order: spending €{safe_quote_amount:.2f} EUR at ask price €{buy_price:.6f} (95% of investable capital, buffer: €{buffer_amount:.2f})")
                success = self.trading_engine.place_order(symbol, 'buy', safe_quote_amount, None)  # Market order with EUR amount
                if success:
                    # Calculate approximate amount purchased (we don't know exact amount until order executes)
                    approx_crypto_amount = safe_quote_amount / buy_price
                    # Record the buy order for position tracking using buy price
                    self.record_buy_order(symbol, approx_crypto_amount, buy_price)
                    logger.info(f"✅ Market buy executed: spent €{safe_quote_amount:.2f} EUR on {symbol} (approx {approx_crypto_amount:.2f} tokens). Position opened.")
                else:
                    logger.warning(f"❌ Failed to buy {symbol}. Will retry next cycle.")
            else:
                logger.info(f"💸 Insufficient funds: €{investable_capital:.2f} available (need ≥€1, buffer: €{buffer_amount:.2f})")
        
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

async def main():
    """Main function to run the bot"""
    bot = CoinbaseSmartAllocationBot()
    await bot.run_smart_allocation_cycle()

if __name__ == "__main__":
    asyncio.run(main()) 