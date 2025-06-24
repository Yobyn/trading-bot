#!/usr/bin/env python3
"""
Enhanced Multi-Asset Trading Bot
Supports multiple portfolios and trading strategies
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger
from config import config
from llm_client import LLMClient
from data_fetcher import DataFetcher
from trading_engine import TradingEngine
from asset_config import get_portfolio, get_strategy, list_available_portfolios, list_available_strategies

class EnhancedMultiAssetBot:
    def __init__(self, portfolio_name: str = "crypto_majors", strategy_name: str = "moderate"):
        self.llm_client = None
        self.data_fetcher = DataFetcher()
        self.trading_engine = TradingEngine()
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
    
    async def analyze_asset(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single asset with enhanced metrics"""
        try:
            # Get market data for this asset
            market_data = await self.data_fetcher.get_market_data(asset["symbol"])
            
            if not market_data or market_data.get('current_price', 0) <= 0:
                return None
            
            # Get LLM decision for this asset
            decision = await self.llm_client.get_trading_decision(market_data)
            
            # Calculate position size based on allocation and strategy
            portfolio_value = self.trading_engine.get_portfolio_summary()['portfolio_value']
            target_position_value = portfolio_value * asset["allocation"]
            max_position_value = portfolio_value * self.strategy["max_position_size"]
            
            # Use the smaller of target allocation or max position size
            actual_position_value = min(target_position_value, max_position_value)
            
            # Calculate quantity to trade
            quantity = actual_position_value / market_data['current_price']
            
            result = {
                'asset': asset,
                'market_data': market_data,
                'decision': decision,
                'position_calculation': {
                    'target_allocation': asset["allocation"],
                    'target_position_value': target_position_value,
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
        """Analyze all assets in the portfolio"""
        results = []
        
        logger.info(f"🔄 Analyzing {len(self.assets)} assets in {self.portfolio_name} portfolio...")
        
        for asset in self.assets:
            try:
                result = await self.analyze_asset(asset)
                
                if result:
                    results.append(result)
                    
                    # Log analysis result
                    market_data = result['market_data']
                    decision = result['decision']
                    position_calc = result['position_calculation']
                    
                    logger.info(
                        f"✅ {asset['name']} ({asset['symbol']}): "
                        f"${market_data['current_price']:.4f} | "
                        f"Decision: {decision['action']} | "
                        f"Allocation: {asset['allocation']*100:.1f}% | "
                        f"Position: ${position_calc['actual_position_value']:.2f}"
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
                        trade_result = await self.trading_engine.execute_trade(
                            decision, 
                            market_data,
                            quantity=position_calc['quantity']
                        )
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
            
            # 1. Analyze all assets
            analysis_results = await self.analyze_all_assets()
            
            if not analysis_results:
                logger.warning("No assets analyzed successfully")
                return
            
            # 2. Execute trades
            await self.execute_trades(analysis_results)
            
            # 3. Log portfolio summary
            portfolio = self.trading_engine.get_portfolio_summary()
            logger.info(f"📊 Portfolio: ${portfolio['portfolio_value']:.2f} | P&L: ${portfolio['total_pnl']:.2f} | Positions: {portfolio['position_count']}")
            
            # 4. Save analysis results
            self.save_analysis_results(analysis_results)
            
        except Exception as e:
            logger.error(f"Error in enhanced trading cycle: {e}")
    
    def save_analysis_results(self, results: List[Dict[str, Any]]):
        """Save analysis results to file"""
        try:
            filename = f"enhanced_analysis_{self.portfolio_name}_{self.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Enhanced analysis results saved to {filename}")
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
    portfolio_name = "crypto_majors"  # Options: crypto_majors, defi_tokens, layer1_blockchains, meme_coins, gaming_tokens, ai_tokens, custom_portfolio
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