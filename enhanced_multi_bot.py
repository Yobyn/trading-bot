#!/usr/bin/env python3
"""
Enhanced Multi-Asset Trading Bot
Advanced trading bot with multiple strategies and portfolio management
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
from config import config
from llm_client import LLMClient
from data_fetcher import DataFetcher
from trading_engine import TradingEngine
from asset_config import get_portfolio, get_strategy, list_available_portfolios, list_available_strategies
from coinbase_smart_allocation_bot import CoinbaseSmartAllocationBot

# Ensure analysis folder exists
ANALYSIS_FOLDER = "analysis"
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

class EnhancedMultiAssetBot:
    def __init__(self, portfolio_name: str = "coinbase_majors", strategy_name: str = "moderate"):
        self.llm_client = None
        self.data_fetcher = DataFetcher()
        self.coinbase_bot = CoinbaseSmartAllocationBot(portfolio_name, strategy_name)
        self.is_running = False
        
        # Load portfolio and strategy
        self.portfolio_name = portfolio_name
        self.strategy_name = strategy_name
        self.assets = get_portfolio(portfolio_name)
        self.strategy = get_strategy(strategy_name)
        
        # Configure logging
        logger.add(
            f"logs/enhanced_multi_bot_{portfolio_name}_{strategy_name}.log",
            rotation="1 day",
            retention="7 days",
            level=config.log_level
        )
        
        logger.info(f"Initialized Enhanced Multi-Asset Bot")
        logger.info(f"Portfolio: {portfolio_name} ({len(self.assets)} assets)")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Max position size: {self.strategy['max_position_size']*100}%")
        logger.info(f"Risk per trade: {self.strategy['risk_per_trade']*100}%")
    
    async def initialize(self):
        """Initialize the enhanced multi-asset trading bot"""
        try:
            logger.info("Initializing Enhanced Multi-Asset Trading Bot...")
            
            # Initialize LLM client
            self.llm_client = LLMClient()
            await self.llm_client.__aenter__()
            
            # Test LLM connection
            test_response = await self.llm_client.generate_response("Hello, are you ready for enhanced multi-asset trading?")
            logger.info(f"LLM test response: {test_response}")
            
            # Test data fetcher
            market_data = await self.data_fetcher.get_market_data()
            logger.info(f"Market data test: {market_data.get('symbol')} @ {market_data.get('current_price')}")
            
            logger.info("Enhanced Multi-Asset Trading Bot initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced multi-asset bot: {e}")
            return False
    
    async def analyze_asset(self, asset: Dict[str, Any], existing_positions: Optional[Dict[str, Any]] = None, phase: str = "MANAGEMENT") -> Optional[Dict[str, Any]]:
        """Analyze a single asset and return trading decision"""
        try:
            # Get comprehensive market data from coinbase bot (includes yearly/weekly averages)
            try:
                market_data = self.coinbase_bot.data_fetcher.get_market_data(asset['symbol'])
                if not market_data:
                    # Fallback to basic data fetcher
                    market_data = await self.data_fetcher.get_market_data(asset['symbol'])
                if not market_data:
                    return None
            except Exception as e:
                logger.warning(f"Could not get comprehensive market data for {asset['symbol']}: {e}")
                # Fallback to basic data fetcher
                market_data = await self.data_fetcher.get_market_data(asset['symbol'])
                if not market_data:
                    return None
            
            # Override portfolio value with actual Coinbase balance if the data is from basic DataFetcher
            if market_data.get('portfolio_value') == 10000.0:  # Basic DataFetcher default
                try:
                    actual_portfolio_value = self.coinbase_bot.trading_engine.get_account_balance()
                    if actual_portfolio_value > 0:
                        market_data['portfolio_value'] = actual_portfolio_value
                        logger.debug(f"💰 {asset['symbol']}: Using actual portfolio value €{actual_portfolio_value:.2f}")
                    else:
                        # Fallback to estimated value based on existing positions
                        total_position_value = sum(pos['eur_value'] for pos in existing_positions.values()) if existing_positions else 100
                        estimated_portfolio = max(total_position_value * 1.5, 100)  # Assume positions are ~66% of portfolio
                        market_data['portfolio_value'] = estimated_portfolio
                        logger.debug(f"📊 {asset['symbol']}: Using estimated portfolio value €{estimated_portfolio:.2f}")
                except Exception as e:
                    logger.warning(f"Could not get portfolio value for {asset['symbol']}: {e}. Using default.")
            
            # Use provided existing positions to avoid multiple API calls
            if existing_positions is None:
                existing_positions = self.coinbase_bot.detect_existing_positions()
            
            current_position = existing_positions.get(asset['symbol'], None)
            
            # Add position information and trading phase to market data for LLM context
            if current_position:
                market_data['current_position'] = {
                    'amount': current_position['amount'],
                    'eur_value': current_position['eur_value'],
                    'buy_price': current_position.get('buy_price'),
                    'profit_loss_pct': current_position.get('profit_loss_pct', 0),
                    'has_position': True
                }
                
                # CRITICAL FIX: Add profit/loss data at top level for LLM client
                market_data['buy_price'] = current_position.get('buy_price')
                market_data['profit_loss_pct'] = current_position.get('profit_loss_pct', 0)
                market_data['profit_loss_eur'] = current_position.get('profit_loss_eur', 0)
                
                # Handle cases where profit_loss_eur might be missing but we can calculate it
                if market_data['profit_loss_eur'] == 0 and market_data['profit_loss_pct'] != 0:
                    # Calculate profit_loss_eur from percentage and position value
                    position_value = current_position['eur_value']
                    profit_loss_pct = market_data['profit_loss_pct']
                    # profit_loss_eur = position_value * (profit_loss_pct / (100 + profit_loss_pct))
                    # This calculates the absolute EUR profit/loss
                    buy_value = position_value / (1 + profit_loss_pct/100)
                    market_data['profit_loss_eur'] = position_value - buy_value
                
                logger.info(f"📊 {asset['symbol']}: Existing position detected - {current_position['amount']:.4f} = €{current_position['eur_value']:.2f}")
                
                # Safe logging with proper error handling
                try:
                    buy_price_val = market_data['buy_price'] if market_data['buy_price'] is not None else 0
                    profit_pct_val = market_data['profit_loss_pct'] if market_data['profit_loss_pct'] is not None else 0
                    profit_eur_val = market_data['profit_loss_eur'] if market_data['profit_loss_eur'] is not None else 0
                    logger.info(f"💰 {asset['symbol']}: Profit/Loss data for LLM - Buy: €{buy_price_val:.6f}, P&L: {profit_pct_val:+.1f}% (€{profit_eur_val:+.2f})")
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not format profit/loss data for {asset['symbol']}: {e}")
                    logger.info(f"💰 {asset['symbol']}: Raw profit/loss data - Buy: {market_data['buy_price']}, P&L: {market_data['profit_loss_pct']}% (€{market_data['profit_loss_eur']})")
            else:
                market_data['current_position'] = {
                    'amount': 0,
                    'eur_value': 0,
                    'buy_price': None,
                    'profit_loss_pct': 0,
                    'has_position': False
                }
                # No top-level profit/loss data for positions we don't own
                logger.info(f"📊 {asset['symbol']}: No existing position")
            
            # Add trading phase context for LLM
            market_data['trading_phase'] = phase
            
            # MINIMUM HOLDING PERIOD CHECK: Prevent selling positions too soon after purchase
            minimum_holding_hours = 4  # 4 hours minimum holding period
            position_age_hours = None
            is_too_new_to_sell = False
            
            if current_position:
                # Get position history to check when it was purchased
                position_history = self.coinbase_bot.load_position_history()
                if asset['symbol'] in position_history:
                    buy_timestamp_str = position_history[asset['symbol']].get('buy_timestamp')
                    if buy_timestamp_str and isinstance(buy_timestamp_str, str):
                        try:
                            buy_timestamp = datetime.fromisoformat(buy_timestamp_str.replace('Z', '+00:00'))
                            now = datetime.now()
                            # Handle timezone-aware vs naive datetime
                            if buy_timestamp.tzinfo is not None and now.tzinfo is None:
                                from datetime import timezone
                                now = now.replace(tzinfo=timezone.utc)
                            elif buy_timestamp.tzinfo is None and now.tzinfo is not None:
                                now = now.replace(tzinfo=None)
                            
                            time_diff = now - buy_timestamp
                            position_age_hours = time_diff.total_seconds() / 3600
                            is_too_new_to_sell = position_age_hours < minimum_holding_hours
                            
                            logger.info(f"📅 {asset['symbol']}: Position age {position_age_hours:.1f} hours (min: {minimum_holding_hours}h) - {'❌ Too new to sell' if is_too_new_to_sell else '✅ Can sell'}")
                        except Exception as e:
                            logger.warning(f"Could not parse buy timestamp for {asset['symbol']}: {e}")
                    else:
                        logger.warning(f"Invalid or missing timestamp for {asset['symbol']}: {buy_timestamp_str}")
            
            # Add holding period context to market data for LLM
            market_data['holding_period_info'] = {
                'minimum_holding_hours': minimum_holding_hours,
                'position_age_hours': position_age_hours,
                'is_too_new_to_sell': is_too_new_to_sell,
                'can_sell': not is_too_new_to_sell if current_position else True
            }
            
            if phase == "INVESTMENT":
                market_data['phase_instruction'] = "INVESTMENT PHASE: I have available cash and I'm looking for the best crypto to invest in. Focus on BUY opportunities."
            else:
                if is_too_new_to_sell:
                    market_data['phase_instruction'] = f"MANAGEMENT PHASE: I'm managing existing positions. IMPORTANT: This position is only {position_age_hours:.1f} hours old (minimum {minimum_holding_hours}h required). You MUST NOT sell positions that are too new - focus on HOLD decisions for recently purchased assets."
                else:
                    market_data['phase_instruction'] = "MANAGEMENT PHASE: I'm managing existing positions. Focus on whether to SELL positions to free up cash for new opportunities, but only if technically justified."
            
            # Get LLM decision for this asset
            if not self.llm_client:
                logger.error(f"LLM client not initialized for {asset['symbol']}")
                return None
            
            logger.info(f"🤖 About to call LLM for {asset['symbol']} with phase: {phase}")
            logger.info(f"🤖 Market data keys: {list(market_data.keys())}")
            logger.info(f"🤖 Current position data: {market_data.get('current_position', {})}")
            
            try:
                # Add timeout for LLM decision to prevent hanging
                logger.info(f"🤖 Calling LLM get_trading_decision for {asset['symbol']}...")
                decision = await asyncio.wait_for(
                    self.llm_client.get_trading_decision(market_data), 
                    timeout=30.0  # 30 second timeout
                )
                logger.info(f"🤖 LLM decision received for {asset['symbol']}: {decision.get('action', 'Unknown')}")
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout for {asset['symbol']}, defaulting to HOLD")
                decision = {
                    'action': 'HOLD',
                    'confidence': 50,
                    'reason': 'LLM timeout - defaulting to HOLD for safety'
                }
            except asyncio.CancelledError:
                logger.debug(f"LLM request cancelled for {asset['symbol']} (shutdown in progress)")
                decision = {
                    'action': 'HOLD',
                    'confidence': 50,
                    'reason': 'Operation cancelled - defaulting to HOLD for safety'
                }
            except Exception as e:
                logger.error(f"LLM error for {asset['symbol']}: {e}, defaulting to HOLD")
                decision = {
                    'action': 'HOLD', 
                    'confidence': 50,
                    'reason': f'LLM error: {str(e)}'
                }
            
            # Get actual portfolio value from Coinbase account
            try:
                portfolio_value = self.coinbase_bot.trading_engine.get_account_balance()
                if portfolio_value == 0:
                    # Fallback to detecting existing positions total value
                    total_value = sum(pos['eur_value'] for pos in existing_positions.values()) if existing_positions else 100  # Minimum €100
                    portfolio_value = max(total_value * 2, 100)  # Assume positions are 50% of total portfolio
                    logger.warning(f"Using estimated portfolio value: €{portfolio_value:.2f}")
            except Exception as e:
                logger.error(f"Could not get account balance: {e}. Using position-based estimate.")
                total_value = sum(pos['eur_value'] for pos in existing_positions.values()) if existing_positions else 100
                portfolio_value = max(total_value * 2, 100)
                logger.warning(f"Using estimated portfolio value: €{portfolio_value:.2f}")
            
            # Check if asset has predefined allocation, otherwise use equal allocation
            if "allocation" in asset:
                asset_allocation = asset["allocation"]
            else:
                # Equal allocation across all assets in portfolio
                asset_allocation = 1.0 / len(self.assets)
            
            target_position_value = portfolio_value * asset_allocation
            max_position_value = portfolio_value * self.strategy["max_position_size"]
            
            # Factor in existing position
            current_position_value = current_position['eur_value'] if current_position else 0
            remaining_target_value = max(0, target_position_value - current_position_value)
            
            # Use the smaller of remaining target or max position size
            actual_position_value = min(remaining_target_value, max_position_value)
            
            # Calculate quantity to trade (only if we need to add to position)
            quantity = actual_position_value / market_data['current_price'] if actual_position_value > 0 else 0
            
            result = {
                'asset': asset,
                'market_data': market_data,
                'decision': decision,
                'current_position': current_position,
                'position_calculation': {
                    'target_allocation': asset_allocation,
                    'target_position_value': target_position_value,
                    'current_position_value': current_position_value,
                    'remaining_target_value': remaining_target_value,
                    'max_position_value': max_position_value,
                    'actual_position_value': actual_position_value,
                    'quantity': quantity,
                    'risk_amount': actual_position_value * self.strategy["risk_per_trade"]
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {asset['symbol']}: {e}")
            return None
    
    async def analyze_investment_opportunity(self, available_cash: float, existing_positions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """INVESTMENT PHASE: Ask LLM to pick the SINGLE best crypto from all options"""
        try:
            logger.info(f"💡 Gathering market data for all {len(self.assets)} cryptos for LLM comparison...")
            
            # Gather market data for ALL cryptos (without individual LLM calls)
            all_crypto_data = []
            for asset in self.assets:
                try:
                    # Skip if we already have a position in this crypto
                    if asset['symbol'] in existing_positions:
                        logger.debug(f"⏭️ Skipping {asset['symbol']} - already have position")
                        continue
                    
                    # Get comprehensive market data
                    market_data = self.coinbase_bot.data_fetcher.get_market_data(asset['symbol'])
                    if not market_data:
                        continue
                        
                    # Add to comparison list
                    crypto_summary = {
                        'symbol': asset['symbol'],
                        'name': asset['name'],
                        'current_price': market_data['current_price'],
                        'three_month_avg': market_data.get('three_month_average', 'Unknown'),
                        'weekly_avg': market_data.get('weekly_average', 'Unknown'),
                        'rsi': market_data.get('rsi', 'Unknown'),
                        'macd': market_data.get('macd', 'Unknown'),
                        'volume_24h': market_data.get('volume_24h', 'Unknown'),
                    }
                    all_crypto_data.append(crypto_summary)
                    
                except Exception as e:
                    logger.warning(f"Could not get data for {asset['symbol']}: {e}")
                    continue
            
            if not all_crypto_data:
                logger.warning("No crypto data available for investment analysis")
                return None
            
            logger.info(f"📊 Gathered data for {len(all_crypto_data)} cryptocurrencies")
            
            # Create consolidated prompt for LLM to pick the BEST crypto
            investment_prompt = f"""You are a technical analysis bot. Based on the market data below, identify which cryptocurrency shows the strongest technical signals for potential upward movement.

Technical Analysis Data for {len(all_crypto_data)} cryptocurrencies:
"""
            
            for crypto in all_crypto_data:
                # Format prices and averages properly (avoid scientific notation)
                current_price = crypto['current_price']
                three_month_avg = crypto['three_month_avg'] if isinstance(crypto['three_month_avg'], (int, float)) else 0
                weekly_avg = crypto['weekly_avg'] if isinstance(crypto['weekly_avg'], (int, float)) else 0
                
                # Calculate explicit price comparisons
                price_vs_3month = "Unknown"
                price_vs_weekly = "Unknown"
                
                if three_month_avg > 0:
                    diff_3month_pct = ((current_price - three_month_avg) / three_month_avg) * 100
                    if diff_3month_pct > 0:
                        price_vs_3month = f"ABOVE 3-month avg by {diff_3month_pct:.1f}% (premium)"
                    else:
                        price_vs_3month = f"BELOW 3-month avg by {abs(diff_3month_pct):.1f}% (discount)"
                        
                if weekly_avg > 0:
                    diff_weekly_pct = ((current_price - weekly_avg) / weekly_avg) * 100
                    if diff_weekly_pct > 0:
                        price_vs_weekly = f"ABOVE weekly avg by {diff_weekly_pct:.1f}% (premium)"
                    else:
                        price_vs_weekly = f"BELOW weekly avg by {abs(diff_weekly_pct):.1f}% (discount)"
                
                investment_prompt += f"""
{crypto['name']} ({crypto['symbol']}):
- Current Price: €{current_price:.6f}
- 3-Month Average: €{three_month_avg:.6f}
- Weekly Average: €{weekly_avg:.6f}
- Current vs 3-Month: {price_vs_3month}
- Current vs Weekly: {price_vs_weekly}
- RSI: {crypto['rsi']}
- MACD: {crypto['macd']}
- Volume 24h: {crypto['volume_24h']}
"""
            
            investment_prompt += f"""

TASK: Analyze the technical indicators (RSI, MACD, price vs averages) and identify which cryptocurrency has the strongest technical setup for potential growth.

Focus on:
- RSI levels (oversold = opportunity)
- MACD signals (positive momentum)
- Current price vs 3-month/weekly averages (undervalued = opportunity)

Respond with ONLY the symbol (e.g., "BTC/EUR") of the cryptocurrency with the best technical setup."""

            # Ask LLM to pick the best crypto
            logger.info(f"🤖 Asking LLM to pick the best crypto from {len(all_crypto_data)} options...")
            
            if not self.llm_client:
                logger.error("LLM client not initialized for investment analysis")
                return None
            
            try:
                llm_response = await asyncio.wait_for(
                    self.llm_client.generate_response(investment_prompt),
                    timeout=60.0  # Longer timeout for complex decision
                )
                
                # Parse LLM response to extract the chosen symbol
                chosen_symbol = self.parse_investment_decision(llm_response, all_crypto_data)
                
                if not chosen_symbol:
                    logger.error("Could not parse LLM investment recommendation")
                    return None
                
                # Find the chosen asset details
                chosen_asset = None
                for asset in self.assets:
                    if asset['symbol'] == chosen_symbol:
                        chosen_asset = asset
                        break
                
                if not chosen_asset:
                    logger.error(f"Could not find asset details for chosen symbol: {chosen_symbol}")
                    return None
                
                logger.info(f"🎯 LLM selected: {chosen_asset['name']} ({chosen_symbol})")
                logger.info(f"💰 Will invest ALL €{available_cash:.2f} into {chosen_asset['name']}")
                
                # Create investment result
                market_data = self.coinbase_bot.data_fetcher.get_market_data(chosen_symbol)
                if not market_data:
                    logger.error(f"Could not get market data for chosen symbol: {chosen_symbol}")
                    return None
                
                investment_result = {
                    'asset': chosen_asset,
                    'market_data': market_data,
                    'decision': {
                        'action': 'BUY',
                        'confidence': 90,
                        'reason': f'LLM selected as best investment from {len(all_crypto_data)} options'
                    },
                    'current_position': None,
                    'position_calculation': {
                        'target_allocation': 1.0,  # Invest all available cash
                        'actual_position_value': available_cash,
                        'quantity': available_cash / market_data['current_price'],
                    },
                    'investment_amount': available_cash,
                    'timestamp': datetime.now().isoformat()
                }
                
                return investment_result
                
            except asyncio.TimeoutError:
                logger.error("LLM investment decision timed out")
                return None
                
        except Exception as e:
            logger.error(f"Error in investment opportunity analysis: {e}")
            return None
    
    def parse_investment_decision(self, llm_response: str, crypto_data: List[Dict]) -> Optional[str]:
        """Parse LLM response to extract the chosen cryptocurrency symbol"""
        try:
            # Extract all symbols that appear in the response
            available_symbols = [crypto['symbol'] for crypto in crypto_data]
            
            # Look for exact symbol matches in the response
            for symbol in available_symbols:
                if symbol in llm_response.upper():
                    return symbol
            
            # Fallback: look for coin names
            for crypto in crypto_data:
                coin_name = crypto['name'].upper()
                if coin_name in llm_response.upper():
                    return crypto['symbol']
            
            logger.warning(f"Could not parse investment decision from: {llm_response}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing investment decision: {e}")
            return None
    
    async def analyze_all_assets(self) -> List[Dict[str, Any]]:
        """Analyze assets based on available cash - Investment Phase vs Management Phase"""
        results = []
        
        # Detect existing positions and available cash
        existing_positions = self.coinbase_bot.detect_existing_positions()
        try:
            # Use direct EUR balance method instead of calculation by subtraction
            # This gets the actual EUR cash balance from Coinbase, excluding crypto holdings
            free_cash = self.coinbase_bot.get_eur_balance()
            
            # Also get total portfolio value for logging
            total_portfolio_value = self.coinbase_bot.trading_engine.get_account_balance()
            total_position_value = sum(pos['eur_value'] for pos in existing_positions.values())
            
            # Log detailed breakdown for transparency
            logger.debug(f"💰 Cash Breakdown: Total Portfolio €{total_portfolio_value:.2f} | Position Values €{total_position_value:.2f} | Direct EUR Cash €{free_cash:.2f}")
            
        except Exception as e:
            logger.warning(f"Could not determine available cash: {e}")
            free_cash = 0
        
        # Apply safety buffer to available cash
        # Keep 5% buffer, but cap at €10 maximum (more conservative for fees, slippage, and rounding)
        buffer_percentage = 0.05  # 5%
        max_buffer = 10.0  # €10 maximum
        
        calculated_buffer = free_cash * buffer_percentage
        safety_buffer = min(calculated_buffer, max_buffer)
        investable_cash = max(0, free_cash - safety_buffer)
        
        logger.debug(f"💰 Buffer Calculation: Free €{free_cash:.2f} | Buffer {buffer_percentage*100}% = €{calculated_buffer:.2f} | Capped at €{safety_buffer:.2f} | Investable €{investable_cash:.2f}")

        # Determine strategy phase
        # CRITICAL FIX: Prevent churning by only going to INVESTMENT PHASE when we have no meaningful positions
        # or when we have substantial cash that represents new deposits
        
        # Calculate meaningful position threshold (positions worth more than €20 are "meaningful")
        meaningful_positions = {k: v for k, v in existing_positions.items() if v['eur_value'] > 20}
        total_meaningful_value = sum(pos['eur_value'] for pos in meaningful_positions.values())
        
        # Only go to INVESTMENT PHASE if:
        # 1. We have substantial cash (>€30) AND no meaningful positions (startup scenario)
        # 2. OR we have very substantial cash (>€50) that likely represents new deposits
        
        if (investable_cash > 30 and len(meaningful_positions) == 0) or (investable_cash > 50):
            # INVESTMENT PHASE: Looking for new investments
            logger.info(f"💰 INVESTMENT PHASE: €{investable_cash:.2f} investable cash available")
            
            if len(meaningful_positions) == 0:
                logger.info(f"🏁 STARTUP SCENARIO: No meaningful positions (>€20), looking for initial investments")
            else:
                logger.info(f"💵 NEW CAPITAL SCENARIO: Substantial cash (>€50) available, likely new deposits")
                logger.info(f"📊 Current meaningful positions: {len(meaningful_positions)} worth €{total_meaningful_value:.2f}")
            
            # Use special investment analysis that picks ONE crypto (but exclude small positions we want to keep)
            investment_result = await self.analyze_investment_opportunity(investable_cash, meaningful_positions)
            if investment_result:
                results.append(investment_result)
            return results
        else:
            # MANAGEMENT PHASE: Focus on existing positions, avoid unnecessary churning
            if existing_positions:
                # Filter assets to only current holdings
                held_symbols = set(existing_positions.keys())
                assets_to_analyze = [asset for asset in self.assets if asset['symbol'] in held_symbols]
                phase = "MANAGEMENT"
                
                # Enhanced logging for management phase
                logger.info(f"📊 MANAGEMENT PHASE: €{investable_cash:.2f} investable cash (insufficient for new investments)")
                logger.info(f"🎯 Managing {len(assets_to_analyze)} existing positions (avoiding churning)")
                logger.info(f"💰 Meaningful positions: {len(meaningful_positions)} worth €{total_meaningful_value:.2f}")
                
                # Log all positions for visibility
                for symbol, pos in existing_positions.items():
                    status = "💰 MEANINGFUL" if pos['eur_value'] > 20 else "🤏 SMALL"
                    profit_indicator = "📈" if pos.get('profit_loss_pct', 0) > 0 else "📉" if pos.get('profit_loss_pct', 0) < 0 else "➡️"
                    logger.info(f"  {status}: {symbol} = €{pos['eur_value']:.2f} {profit_indicator} {pos.get('profit_loss_pct', 0):+.1f}%")
                
                # DEBUG: Log the filtering process
                logger.info(f"🔍 MANAGEMENT DEBUG: Held symbols: {held_symbols}")
                logger.info(f"🔍 MANAGEMENT DEBUG: Portfolio has {len(self.assets)} total assets")
                logger.info(f"🔍 MANAGEMENT DEBUG: Assets to analyze: {[asset['symbol'] for asset in assets_to_analyze]}")
                
            else:
                # No positions and insufficient cash - wait for more capital
                logger.info(f"⏸️ WAITING: €{investable_cash:.2f} insufficient for new investments (need >€30)")
                logger.info(f"💡 Suggestion: Add more funds or wait for market movements")
                return results  # Return empty results, don't analyze anything
        
        logger.info(f"📊 Found {len(existing_positions)} existing positions | Free Cash: €{free_cash:.2f} | Investable: €{investable_cash:.2f} (after €{safety_buffer:.2f} buffer)")
        
        for asset in assets_to_analyze:
            try:
                logger.info(f"🔍 Starting analysis for {asset['symbol']} in {phase} phase...")
                result = await self.analyze_asset(asset, existing_positions, phase)
                
                if result:
                    results.append(result)
                    
                    # Log analysis result
                    market_data = result['market_data']
                    decision = result['decision']
                    position_calc = result['position_calculation']
                    
                    # Show position status
                    current_pos = result.get('current_position')
                    position_status = "No Position"
                    if current_pos and current_pos['amount'] > 0:
                        profit_loss = current_pos.get('profit_loss_pct', 0)
                        profit_indicator = "📈" if profit_loss > 0 else "📉" if profit_loss < 0 else "➡️"
                        position_status = f"€{current_pos['eur_value']:.2f} {profit_indicator} {profit_loss:+.1f}%"
                    
                    logger.info(
                        f"✅ {asset['name']} ({asset['symbol']}): "
                        f"€{market_data['current_price']:.4f} | "
                        f"Decision: {decision['action']} | "
                        f"Current: {position_status} | "
                        f"Target: {position_calc['target_allocation']*100:.1f}% | "
                        f"Action: €{position_calc['actual_position_value']:.2f}"
                    )
                else:
                    logger.warning(f"❌ Failed to analyze {asset['symbol']} - result was None")
                    
            except Exception as e:
                logger.error(f"Error analyzing {asset['symbol']}: {e}", exc_info=True)
        
        return results
    
    async def execute_trades(self, analysis_results: List[Dict[str, Any]]):
        """Execute trades based on analysis results with strategy constraints"""
        for result in analysis_results:
            try:
                asset = result['asset']
                market_data = result['market_data']
                decision = result['decision']
                position_calc = result['position_calculation']
                
                # Apply strategy constraints
                if decision['action'] == 'BUY':
                    # Check if this is an investment phase result (has investment_amount)
                    if 'investment_amount' in result:
                        # INVESTMENT PHASE: Invest ALL available cash into chosen crypto
                        eur_amount = result['investment_amount']
                        symbol = asset['symbol']
                        
                        # MINIMUM INVESTMENT CHECK: Don't invest less than €1 (not worth the fees)
                        if eur_amount < 1.0:
                            logger.warning(f"💰 MINIMUM INVESTMENT BLOCK: Cannot invest in {asset['name']} - amount €{eur_amount:.2f} is below €1.00 minimum")
                            logger.info(f"💡 Investment too small to be profitable (fees would exceed value)")
                            continue  # Skip this buy order
                        
                        logger.info(f"🎯 INVESTMENT PHASE: Investing ALL €{eur_amount:.2f} into {asset['name']}")
                        
                        # Get buy price for position tracking
                        buy_price = self.coinbase_bot.get_buy_price(result['market_data'])
                        approx_crypto_amount = eur_amount / buy_price
                        
                        # Use coinbase trading engine which respects paper trading config
                        success = self.coinbase_bot.trading_engine.place_order(symbol, 'buy', eur_amount, None)
                        if success:
                            # CRITICAL FIX: Record the buy order for holding period tracking
                            self.coinbase_bot.record_buy_order(symbol, approx_crypto_amount, buy_price)
                            logger.info(f"✅ Successfully invested ALL cash into {asset['name']}")
                            logger.info(f"📝 Recorded purchase: {approx_crypto_amount:.6f} {symbol} at €{buy_price:.6f} for holding period tracking")
                        else:
                            logger.error(f"❌ Failed to invest in {asset['name']}")
                    else:
                        # REGULAR BUY: Check if we have enough capital for position sizing
                        if position_calc['actual_position_value'] > 0:
                            eur_amount = position_calc['actual_position_value']
                            symbol = asset['symbol']
                            
                            # MINIMUM INVESTMENT CHECK: Don't invest less than €1 (not worth the fees)
                            if eur_amount < 1.0:
                                logger.warning(f"💰 MINIMUM INVESTMENT BLOCK: Cannot invest in {asset['name']} - amount €{eur_amount:.2f} is below €1.00 minimum")
                                logger.info(f"💡 Investment too small to be profitable (fees would exceed value)")
                                continue  # Skip this buy order
                            
                            logger.info(f"💸 Investing in {asset['name']}: €{eur_amount:.2f}")
                            
                            # Get buy price for position tracking
                            buy_price = self.coinbase_bot.get_buy_price(result['market_data'])
                            approx_crypto_amount = eur_amount / buy_price
                            
                            # Use coinbase trading engine which respects paper trading config
                            success = self.coinbase_bot.trading_engine.place_order(symbol, 'buy', eur_amount, None)
                            if success:
                                # CRITICAL FIX: Record the buy order for holding period tracking
                                self.coinbase_bot.record_buy_order(symbol, approx_crypto_amount, buy_price)
                                logger.info(f"✅ Successfully invested in {asset['name']}")
                                logger.info(f"📝 Recorded purchase: {approx_crypto_amount:.6f} {symbol} at €{buy_price:.6f} for holding period tracking")
                            else:
                                logger.error(f"❌ Failed to invest in {asset['name']}")
                        else:
                            logger.warning(f"Insufficient capital for {asset['name']}")
                
                elif decision['action'] == 'SELL':
                    # Check if we have an existing position to close using the correct position data
                    position_data = result['market_data']['current_position']
                    raw_position = result.get('current_position')
                    
                    # POSITION STABILITY: Check minimum holding period before selling
                    holding_period_info = result['market_data'].get('holding_period_info', {})
                    is_too_new_to_sell = holding_period_info.get('is_too_new_to_sell', False)
                    position_age_hours = holding_period_info.get('position_age_hours')
                    minimum_holding_hours = holding_period_info.get('minimum_holding_hours', 4)
                    
                    if is_too_new_to_sell:
                        logger.warning(f"🔒 HOLDING PERIOD BLOCK: Cannot sell {asset['name']} - position is only {position_age_hours:.1f} hours old (minimum: {minimum_holding_hours}h)")
                        logger.info(f"⏰ Position must be held for {minimum_holding_hours - position_age_hours:.1f} more hours before selling")
                        continue  # Skip this sell order
                    
                    if position_data and position_data.get('has_position', False) and raw_position:
                        # MINIMUM POSITION VALUE CHECK: Don't sell positions worth less than €1 (not worth the fees)
                        position_value = raw_position['eur_value']
                        if position_value < 1.0:
                            logger.warning(f"💰 MINIMUM VALUE BLOCK: Cannot sell {asset['name']} - position value €{position_value:.2f} is below €1.00 minimum")
                            logger.info(f"💡 Position too small to sell profitably (fees would exceed value)")
                            continue  # Skip this sell order
                        
                        # SELL = liquidate existing position (use coinbase trading engine for proper paper trading)
                        amount = raw_position['amount']
                        symbol = asset['symbol']
                        
                        # Additional stability check: Don't sell profitable positions unless strongly justified
                        profit_loss_pct = raw_position.get('profit_loss_pct', 0)
                        if profit_loss_pct > 5:  # Position is profitable by more than 5%
                            logger.info(f"💎 PROFITABLE POSITION: {asset['name']} is +{profit_loss_pct:.1f}% profitable - selling based on LLM technical analysis")
                        
                        logger.info(f"💰 Liquidating {asset['name']}: {amount:.6f} tokens = €{position_value:.2f} (held {position_age_hours:.1f}h)")
                        
                        # Use coinbase trading engine which respects paper trading config
                        success = self.coinbase_bot.trading_engine.place_order(symbol, 'sell', amount, None)
                        if success:
                            logger.info(f"✅ Successfully liquidated {asset['name']}")
                        else:
                            logger.error(f"❌ Failed to liquidate {asset['name']}")
                    else:
                        # This shouldn't happen in MANAGEMENT PHASE - log warning
                        logger.warning(f"⚠️ SELL decision for {asset['name']} but no existing position found. Skipping.")
                
                else:  # HOLD
                    logger.info(f"Holding {asset['name']} - no action taken")
                
            except Exception as e:
                logger.error(f"Error executing trade for {asset['symbol']}: {e}")
    
    async def run_trading_cycle(self):
        """Run one complete enhanced trading cycle"""
        try:
            logger.info("🔄 Starting enhanced multi-asset trading cycle...")
            
            # 0. Clean up position history (remove entries for positions no longer held)
            self.coinbase_bot.cleanup_position_history()
            
            # 1. Analyze all assets
            analysis_results = await self.analyze_all_assets()
            
            if not analysis_results:
                logger.warning("No assets analyzed successfully")
                return
            
            # 2. Execute trades
            await self.execute_trades(analysis_results)
            
            # 3. Log portfolio summary
            try:
                # Get actual portfolio value from Coinbase
                portfolio_value = self.coinbase_bot.trading_engine.get_account_balance()
                existing_positions = self.coinbase_bot.detect_existing_positions()
                total_position_value = sum(pos['eur_value'] for pos in existing_positions.values())
                total_pnl = sum(pos.get('profit_loss_eur', 0) for pos in existing_positions.values())
                
                logger.info(f"📊 Portfolio: €{portfolio_value:.2f} | Positions Value: €{total_position_value:.2f} | P&L: €{total_pnl:.2f} | Assets: {len(existing_positions)}")
            except Exception as e:
                logger.warning(f"Could not get portfolio summary: {e}")
                logger.info(f"📊 Portfolio: Unable to retrieve portfolio data")
            
            # 4. Save analysis results
            self.save_analysis_results(analysis_results)
            
        except asyncio.CancelledError:
            logger.debug("Trading cycle cancelled (shutdown in progress)")
            # Don't log as error - this is expected during shutdown
            pass
        except Exception as e:
            logger.error(f"Error in enhanced trading cycle: {e}")
    
    def save_analysis_results(self, results: List[Dict[str, Any]]):
        """Save analysis results to the analysis folder"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_analysis_{self.portfolio_name}_{self.strategy_name}_{timestamp}.json"
            filepath = os.path.join(ANALYSIS_FOLDER, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Enhanced analysis results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
    
    async def start(self, interval_minutes: int = 15):
        """Start the enhanced multi-asset trading bot"""
        if not await self.initialize():
            logger.error("Failed to initialize enhanced multi-asset bot")
            return
        
        self.is_running = True
        logger.info(f"🚀 Starting enhanced multi-asset trading bot")
        logger.info(f"📈 Portfolio: {self.portfolio_name} ({len(self.assets)} assets)")
        logger.info(f"⚡ Strategy: {self.strategy_name}")
        logger.info(f"⏰ Interval: {interval_minutes} minutes")
        
        try:
            while self.is_running:
                await self.run_trading_cycle()
                
                # Wait for next cycle
                logger.info(f"⏰ Waiting {interval_minutes} minutes until next trading cycle...")
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping bot...")
        except asyncio.CancelledError:
            logger.info("Bot operation cancelled, stopping...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the enhanced multi-asset trading bot"""
        self.is_running = False
        
        if self.llm_client:
            try:
                await self.llm_client.__aexit__(None, None, None)
            except (asyncio.CancelledError, Exception) as e:
                # Ignore cleanup errors during shutdown
                logger.debug(f"LLM client cleanup error (expected during shutdown): {e}")
                pass
        
        logger.info("Enhanced multi-asset trading bot stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        try:
            # Try to get coinbase portfolio data
            portfolio_value = self.coinbase_bot.trading_engine.get_account_balance()
            existing_positions = self.coinbase_bot.detect_existing_positions()
            portfolio = {
                'portfolio_value': portfolio_value,
                'position_count': len(existing_positions),
                'total_pnl': sum(pos.get('profit_loss_eur', 0) for pos in existing_positions.values())
            }
        except:
            # Fallback portfolio data
            portfolio = {
                'portfolio_value': 0,
                'position_count': 0,
                'total_pnl': 0
            }
        
        return {
            'is_running': self.is_running,
            'portfolio_name': self.portfolio_name,
            'strategy_name': self.strategy_name,
            'assets_monitored': len(self.assets),
            'asset_list': [asset['symbol'] for asset in self.assets],
            'portfolio': portfolio,
            'strategy': self.strategy,
            'config': {
                'trading_enabled': config.trading_enabled,
                'paper_trading': config.paper_trading,
                'llm_model': config.llm_model
            }
        }

async def main():
    """Main function to run the enhanced multi-asset trading bot"""
    # You can change these parameters
    portfolio_name = "coinbase_all_eur"  # Options: coinbase_majors, coinbase_all_eur, coinbase_majors_usd
    strategy_name = "moderate"        # Options: conservative, moderate, aggressive, scalping
    
    bot = EnhancedMultiAssetBot(portfolio_name=portfolio_name, strategy_name=strategy_name)
    
    # Run with 15 minute intervals
    await bot.start(interval_minutes=15)

if __name__ == "__main__":
    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Run the bot
    asyncio.run(main()) 