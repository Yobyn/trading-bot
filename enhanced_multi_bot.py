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
        self.trading_engine = TradingEngine()
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
                logger.info(f"📊 {asset['symbol']}: Existing position detected - {current_position['amount']:.4f} = €{current_position['eur_value']:.2f}")
            else:
                market_data['current_position'] = {
                    'amount': 0,
                    'eur_value': 0,
                    'buy_price': None,
                    'profit_loss_pct': 0,
                    'has_position': False
                }
                logger.info(f"📊 {asset['symbol']}: No existing position")
            
            # Add trading phase context for LLM
            market_data['trading_phase'] = phase
            if phase == "INVESTMENT":
                market_data['phase_instruction'] = "INVESTMENT PHASE: I have available cash and I'm looking for the best crypto to invest in. Focus on BUY opportunities."
            else:
                market_data['phase_instruction'] = "MANAGEMENT PHASE: I'm managing existing positions. Focus on whether to SELL positions to free up cash for new opportunities."
            
            # Get LLM decision for this asset
            if not self.llm_client:
                logger.error(f"LLM client not initialized for {asset['symbol']}")
                return None
            
            try:
                # Add timeout for LLM decision to prevent hanging
                decision = await asyncio.wait_for(
                    self.llm_client.get_trading_decision(market_data), 
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout for {asset['symbol']}, defaulting to HOLD")
                decision = {
                    'action': 'HOLD',
                    'confidence': 50,
                    'reason': 'LLM timeout - defaulting to HOLD for safety'
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
    
    async def analyze_all_assets(self) -> List[Dict[str, Any]]:
        """Analyze assets based on available cash - Investment Phase vs Management Phase"""
        results = []
        
        # Detect existing positions and available cash
        existing_positions = self.coinbase_bot.detect_existing_positions()
        try:
            available_cash = self.coinbase_bot.trading_engine.get_account_balance()
            # Subtract current position values to get free cash
            total_position_value = sum(pos['eur_value'] for pos in existing_positions.values())
            free_cash = available_cash - total_position_value
        except Exception as e:
            logger.warning(f"Could not determine available cash: {e}")
            free_cash = 0
        
        # Determine strategy phase
        if free_cash > 10 and len(existing_positions) > 0:
            # INVESTMENT PHASE: Analyze all 41 cryptos to find best investment opportunity
            assets_to_analyze = self.assets
            phase = "INVESTMENT"
            logger.info(f"💰 INVESTMENT PHASE: €{free_cash:.2f} available cash > €10")
            logger.info(f"🔍 Analyzing ALL {len(assets_to_analyze)} cryptos to find best investment opportunity...")
        else:
            # MANAGEMENT PHASE: Only analyze current holdings for potential sales
            if existing_positions:
                # Filter assets to only current holdings
                held_symbols = set(existing_positions.keys())
                assets_to_analyze = [asset for asset in self.assets if asset['symbol'] in held_symbols]
                phase = "MANAGEMENT"
                logger.info(f"📊 MANAGEMENT PHASE: €{free_cash:.2f} available cash (≤ €10 or no positions)")
                logger.info(f"🎯 Analyzing ONLY {len(assets_to_analyze)} current holdings for potential sales...")
            else:
                # No positions and no cash - analyze all for potential investment
                assets_to_analyze = self.assets
                phase = "INVESTMENT"
                logger.info(f"🏁 STARTUP: No positions found, analyzing all {len(assets_to_analyze)} cryptos...")
        
        logger.info(f"📊 Found {len(existing_positions)} existing positions | Available Cash: €{free_cash:.2f}")
        
        for asset in assets_to_analyze:
            try:
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
                    logger.warning(f"❌ Failed to analyze {asset['symbol']}")
                    
            except Exception as e:
                logger.error(f"Error analyzing {asset['symbol']}: {e}")
        
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
                    # Check if we have enough capital
                    if position_calc['actual_position_value'] > 0:
                        # Execute trade with calculated position size
                        trade_result = await self.trading_engine.execute_trade(decision, market_data)
                        logger.info(f"Trade Result for {asset['name']}: {trade_result['status']}")
                    else:
                        logger.warning(f"Insufficient capital for {asset['name']}")
                
                elif decision['action'] == 'SELL':
                    # Execute sell order
                    trade_result = await self.trading_engine.execute_trade(decision, market_data)
                    logger.info(f"Trade Result for {asset['name']}: {trade_result['status']}")
                
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
                portfolio = self.trading_engine.get_portfolio_summary()
                logger.info(f"📊 Portfolio (Paper): ${portfolio['portfolio_value']:.2f} | P&L: ${portfolio['total_pnl']:.2f} | Positions: {portfolio['position_count']}")
            
            # 4. Save analysis results
            self.save_analysis_results(analysis_results)
            
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
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping bot...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the enhanced multi-asset trading bot"""
        self.is_running = False
        
        if self.llm_client:
            await self.llm_client.__aexit__(None, None, None)
        
        logger.info("Enhanced multi-asset trading bot stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        portfolio = self.trading_engine.get_portfolio_summary()
        
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