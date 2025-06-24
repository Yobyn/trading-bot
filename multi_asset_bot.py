#!/usr/bin/env python3
"""
Multi-Asset Trading Bot
Trades multiple assets simultaneously with AI decisions
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

class MultiAssetTradingBot:
    def __init__(self):
        self.llm_client = None
        self.data_fetcher = DataFetcher()
        self.trading_engine = TradingEngine()
        self.is_running = False
        
        # Define multiple assets to trade
        self.assets = [
            # Cryptocurrencies
            {"symbol": "BTC/USDT", "name": "Bitcoin", "exchange": "binance"},
            {"symbol": "ETH/USDT", "name": "Ethereum", "exchange": "binance"},
            {"symbol": "ADA/USDT", "name": "Cardano", "exchange": "binance"},
            {"symbol": "SOL/USDT", "name": "Solana", "exchange": "binance"},
            {"symbol": "DOT/USDT", "name": "Polkadot", "exchange": "binance"},
            
            # Popular Altcoins
            {"symbol": "MATIC/USDT", "name": "Polygon", "exchange": "binance"},
            {"symbol": "LINK/USDT", "name": "Chainlink", "exchange": "binance"},
            {"symbol": "UNI/USDT", "name": "Uniswap", "exchange": "binance"},
            {"symbol": "AVAX/USDT", "name": "Avalanche", "exchange": "binance"},
            {"symbol": "ATOM/USDT", "name": "Cosmos", "exchange": "binance"},
        ]
        
        # Portfolio allocation per asset (1% each = 10% total)
        self.asset_allocation = 0.01  # 1% per asset
        
        # Configure logging
        logger.add(
            "logs/multi_asset_bot.log",
            rotation="1 day",
            retention="7 days",
            level=config.log_level
        )
    
    async def initialize(self):
        """Initialize the multi-asset trading bot"""
        try:
            logger.info("Initializing Multi-Asset Trading Bot...")
            
            # Initialize LLM client
            self.llm_client = LLMClient()
            await self.llm_client.__aenter__()
            
            # Test LLM connection
            test_response = await self.llm_client.generate_response("Hello, are you ready for multi-asset trading?")
            logger.info(f"LLM test response: {test_response}")
            
            # Test data fetcher
            market_data = await self.data_fetcher.get_market_data()
            logger.info(f"Market data test: {market_data.get('symbol')} @ {market_data.get('current_price')}")
            
            logger.info("Multi-Asset Trading Bot initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-asset bot: {e}")
            return False
    
    async def analyze_all_assets(self) -> List[Dict[str, Any]]:
        """Analyze all assets and get market data"""
        results = []
        
        for asset in self.assets:
            try:
                # Get market data for this asset
                market_data = await self.data_fetcher.get_market_data(asset["symbol"])
                
                if market_data and market_data.get('current_price', 0) > 0:
                    # Get LLM decision for this asset
                    decision = await self.llm_client.get_trading_decision(market_data)
                    
                    result = {
                        'asset': asset,
                        'market_data': market_data,
                        'decision': decision,
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    logger.info(f"✅ {asset['name']} ({asset['symbol']}): ${market_data['current_price']:.2f} | Decision: {decision['action']}")
                else:
                    logger.warning(f"❌ Failed to get data for {asset['symbol']}")
                    
            except Exception as e:
                logger.error(f"Error analyzing {asset['symbol']}: {e}")
        
        return results
    
    async def execute_trades(self, analysis_results: List[Dict[str, Any]]):
        """Execute trades based on analysis results"""
        for result in analysis_results:
            try:
                asset = result['asset']
                market_data = result['market_data']
                decision = result['decision']
                
                # Execute trade for this asset
                trade_result = await self.trading_engine.execute_trade(decision, market_data)
                
                logger.info(f"Trade Result for {asset['name']}: {trade_result['status']}")
                
            except Exception as e:
                logger.error(f"Error executing trade for {asset['symbol']}: {e}")
    
    async def run_trading_cycle(self):
        """Run one complete multi-asset trading cycle"""
        try:
            logger.info("🔄 Starting multi-asset trading cycle...")
            
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
            logger.error(f"Error in multi-asset trading cycle: {e}")
    
    def save_analysis_results(self, results: List[Dict[str, Any]]):
        """Save analysis results to file"""
        try:
            filename = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Analysis results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
    
    async def start(self, interval_minutes: int = 15):
        """Start the multi-asset trading bot"""
        if not await self.initialize():
            logger.error("Failed to initialize multi-asset bot")
            return
        
        self.is_running = True
        logger.info(f"🚀 Starting multi-asset trading bot with {interval_minutes} minute intervals")
        logger.info(f"📈 Monitoring {len(self.assets)} assets")
        
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
        """Stop the multi-asset trading bot"""
        self.is_running = False
        
        if self.llm_client:
            await self.llm_client.__aexit__(None, None, None)
        
        logger.info("Multi-asset trading bot stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        portfolio = self.trading_engine.get_portfolio_summary()
        
        return {
            'is_running': self.is_running,
            'assets_monitored': len(self.assets),
            'asset_list': [asset['symbol'] for asset in self.assets],
            'portfolio': portfolio,
            'config': {
                'trading_enabled': config.trading_enabled,
                'paper_trading': config.paper_trading,
                'asset_allocation': self.asset_allocation,
                'llm_model': config.llm_model
            }
        }

async def main():
    """Main function to run the multi-asset trading bot"""
    bot = MultiAssetTradingBot()
    
    # Run with 15 minute intervals
    await bot.start(interval_minutes=15)

if __name__ == "__main__":
    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Run the bot
    asyncio.run(main()) 