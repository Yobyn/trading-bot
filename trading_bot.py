import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from loguru import logger
import json

from config import config
from llm_client import LLMClient
from data_fetcher import DataFetcher
from trading_engine import TradingEngine

class TradingBot:
    def __init__(self):
        self.llm_client = None
        self.data_fetcher = DataFetcher()
        self.trading_engine = TradingEngine()
        self.is_running = False
        self.last_decision = None
        self.trade_history = []
        
        # Configure logging
        logger.add(
            "logs/trading_bot.log",
            rotation="1 day",
            retention="7 days",
            level=config.log_level
        )
        
    async def initialize(self):
        """Initialize the trading bot"""
        try:
            logger.info("Initializing Trading Bot...")
            
            # Initialize LLM client
            self.llm_client = LLMClient()
            await self.llm_client.__aenter__()
            
            # Test LLM connection
            test_response = await self.llm_client.generate_response("Hello, are you ready for trading?")
            logger.info(f"LLM test response: {test_response}")
            
            # Test data fetcher
            market_data = await self.data_fetcher.get_market_data()
            logger.info(f"Market data test: {market_data.get('symbol')} @ {market_data.get('current_price')}")
            
            logger.info("Trading Bot initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            return False
    
    async def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            logger.info("Starting trading cycle...")
            
            # 1. Fetch market data
            market_data = await self.data_fetcher.get_market_data()
            if not market_data or market_data.get('current_price', 0) == 0:
                logger.warning("Failed to fetch valid market data")
                return
            
            logger.info(f"Market data: {market_data['symbol']} @ {market_data['current_price']}")
            
            # 2. Check stop losses first
            closed_positions = await self.trading_engine.check_stop_losses(market_data)
            if closed_positions:
                logger.info(f"Closed positions due to stop loss/take profit: {closed_positions}")
            
            # 3. Get LLM decision
            decision = await self.llm_client.get_trading_decision(market_data)
            self.last_decision = decision
            
            logger.info(f"LLM Decision: {decision['action']} - {decision['reason']}")
            
            # 4. Execute trade
            trade_result = await self.trading_engine.execute_trade(decision, market_data)
            
            # 5. Log trade result
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'market_data': market_data,
                'decision': decision,
                'trade_result': trade_result
            }
            self.trade_history.append(trade_record)
            
            # Keep only last 100 trades
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
            
            logger.info(f"Trade result: {trade_result['status']}")
            
            # 6. Log portfolio summary
            portfolio = self.trading_engine.get_portfolio_summary()
            logger.info(f"Portfolio: ${portfolio['portfolio_value']:.2f} | P&L: ${portfolio['total_pnl']:.2f} | Positions: {portfolio['position_count']}")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def start(self, interval_minutes: int = 60):
        """Start the trading bot"""
        if not await self.initialize():
            logger.error("Failed to initialize trading bot")
            return
        
        self.is_running = True
        logger.info(f"Starting trading bot with {interval_minutes} minute intervals")
        
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
        """Stop the trading bot"""
        self.is_running = False
        
        if self.llm_client:
            await self.llm_client.__aexit__(None, None, None)
        
        logger.info("Trading bot stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        portfolio = self.trading_engine.get_portfolio_summary()
        
        return {
            'is_running': self.is_running,
            'last_decision': self.last_decision,
            'portfolio': portfolio,
            'trade_count': len(self.trade_history),
            'config': {
                'trading_enabled': config.trading_enabled,
                'paper_trading': config.paper_trading,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'llm_model': config.llm_model
            }
        }
    
    def save_trade_history(self, filename: str = None):
        """Save trade history to file"""
        if not filename:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
            logger.info(f"Trade history saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save trade history: {e}")
    
    def load_trade_history(self, filename: str):
        """Load trade history from file"""
        try:
            with open(filename, 'r') as f:
                self.trade_history = json.load(f)
            logger.info(f"Trade history loaded from {filename}")
        except Exception as e:
            logger.error(f"Failed to load trade history: {e}")

async def main():
    """Main function to run the trading bot"""
    bot = TradingBot()
    
    # Run with 1 hour intervals
    await bot.start(interval_minutes=60)

if __name__ == "__main__":
    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Run the bot
    asyncio.run(main()) 