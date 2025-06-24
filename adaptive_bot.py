#!/usr/bin/env python3
"""
Adaptive Multi-Asset Trading Bot
Automatically adjusts strategy based on performance and market conditions
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
from config import config
from llm_client import LLMClient
from data_fetcher import DataFetcher
from trading_engine import TradingEngine
from asset_config import get_portfolio, get_strategy, list_available_strategies

class AdaptiveMultiAssetBot:
    def __init__(self, portfolio_name: str = "crypto_majors"):
        self.llm_client = None
        self.data_fetcher = DataFetcher()
        self.trading_engine = TradingEngine()
        self.is_running = False
        
        # Portfolio configuration
        self.portfolio_name = portfolio_name
        self.assets = get_portfolio(portfolio_name)
        
        # Adaptive strategy management
        self.current_strategy = "moderate"
        self.strategy_performance = {}
        self.performance_history = []
        self.market_conditions = {}
        
        # Performance thresholds for strategy adjustment
        self.performance_thresholds = {
            "excellent": 0.05,    # 5% daily return
            "good": 0.02,         # 2% daily return
            "poor": -0.02,        # -2% daily return
            "terrible": -0.05     # -5% daily return
        }
        
        # Strategy adjustment rules
        self.strategy_adjustments = {
            "excellent": "aggressive",    # Increase risk when doing well
            "good": "moderate",           # Stay moderate when doing okay
            "poor": "conservative",       # Reduce risk when losing
            "terrible": "scalping"        # Very conservative when doing badly
        }
        
        # Market condition thresholds
        self.market_thresholds = {
            "high_volatility": 0.05,     # 5% daily volatility
            "low_volatility": 0.01,      # 1% daily volatility
            "bull_market": 0.03,         # 3% daily gain
            "bear_market": -0.03         # -3% daily loss
        }
        
        # Load current strategy
        self.strategy = get_strategy(self.current_strategy)
        
        # Configure logging
        logger.add(
            f"logs/adaptive_bot_{portfolio_name}.log",
            rotation="1 day",
            retention="7 days",
            level=config.log_level
        )
        
        logger.info(f"Initialized Adaptive Multi-Asset Bot")
        logger.info(f"Portfolio: {portfolio_name} ({len(self.assets)} assets)")
        logger.info(f"Initial Strategy: {self.current_strategy}")
    
    async def initialize(self):
        """Initialize the adaptive multi-asset trading bot"""
        try:
            logger.info("Initializing Adaptive Multi-Asset Trading Bot...")
            
            # Initialize LLM client
            self.llm_client = LLMClient()
            await self.llm_client.__aenter__()
            
            # Test LLM connection
            test_response = await self.llm_client.generate_response("Hello, are you ready for adaptive multi-asset trading?")
            logger.info(f"LLM test response: {test_response}")
            
            # Test data fetcher
            market_data = await self.data_fetcher.get_market_data()
            logger.info(f"Market data test: {market_data.get('symbol')} @ {market_data.get('current_price')}")
            
            logger.info("Adaptive Multi-Asset Trading Bot initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize adaptive bot: {e}")
            return False
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics"""
        try:
            portfolio = self.trading_engine.get_portfolio_summary()
            
            # Calculate daily return
            daily_pnl = portfolio.get('daily_pnl', 0)
            portfolio_value = portfolio.get('portfolio_value', 10000)
            daily_return = daily_pnl / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate volatility (if we have enough history)
            if len(self.performance_history) >= 7:
                returns = [p.get('daily_return', 0) for p in self.performance_history[-7:]]
                volatility = np.std(returns) if len(returns) > 1 else 0.01
            else:
                volatility = 0.01  # Default low volatility
            
            # Calculate win rate
            total_trades = len(portfolio.get('orders', []))
            winning_trades = len([o for o in portfolio.get('orders', []) if o.get('pnl', 0) > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.5
            
            # Calculate Sharpe ratio (simplified)
            if len(self.performance_history) >= 7:
                returns = [p.get('daily_return', 0) for p in self.performance_history[-7:]]
                avg_return = np.mean(returns) if len(returns) > 0 else 0
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            else:
                sharpe_ratio = 0
            
            metrics = {
                'daily_return': daily_return,
                'volatility': volatility,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'portfolio_value': portfolio_value,
                'total_pnl': portfolio.get('total_pnl', 0),
                'daily_pnl': daily_pnl,
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            # Return default metrics if calculation fails
            return {
                'daily_return': 0.0,
                'volatility': 0.01,
                'win_rate': 0.5,
                'sharpe_ratio': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'portfolio_value': 10000.0,
                'total_pnl': 0.0,
                'daily_pnl': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    async def assess_market_conditions(self) -> Dict[str, Any]:
        """Assess current market conditions"""
        try:
            # Get market data for major assets to assess overall market
            btc_data = await self.data_fetcher.get_market_data("BTC/USDT")
            eth_data = await self.data_fetcher.get_market_data("ETH/USDT")
            
            # Calculate market volatility
            btc_change = abs(btc_data.get('price_change_24h', 0) / btc_data.get('current_price', 1))
            eth_change = abs(eth_data.get('price_change_24h', 0) / eth_data.get('current_price', 1))
            market_volatility = (btc_change + eth_change) / 2
            
            # Determine market trend
            btc_trend = btc_data.get('price_change_24h', 0) / btc_data.get('current_price', 1)
            eth_trend = eth_data.get('price_change_24h', 0) / eth_data.get('current_price', 1)
            market_trend = (btc_trend + eth_trend) / 2
            
            # Classify market conditions
            if market_volatility > self.market_thresholds['high_volatility']:
                volatility_state = "high_volatility"
            elif market_volatility < self.market_thresholds['low_volatility']:
                volatility_state = "low_volatility"
            else:
                volatility_state = "normal_volatility"
            
            if market_trend > self.market_thresholds['bull_market']:
                trend_state = "bull_market"
            elif market_trend < self.market_thresholds['bear_market']:
                trend_state = "bear_market"
            else:
                trend_state = "sideways_market"
            
            conditions = {
                'volatility': volatility_state,
                'trend': trend_state,
                'market_volatility': market_volatility,
                'market_trend': market_trend,
                'btc_price': btc_data.get('current_price', 0),
                'eth_price': eth_data.get('current_price', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return {
                'volatility': 'normal_volatility',
                'trend': 'sideways_market',
                'market_volatility': 0.02,
                'market_trend': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def determine_optimal_strategy(self, performance_metrics: Dict[str, Any], market_conditions: Dict[str, Any]) -> str:
        """Determine the optimal strategy based on performance and market conditions"""
        daily_return = performance_metrics['daily_return']
        volatility = performance_metrics['volatility']
        win_rate = performance_metrics['win_rate']
        market_volatility = market_conditions['market_volatility']
        market_trend = market_conditions['market_trend']
        
        # Performance-based strategy selection
        if daily_return > self.performance_thresholds['excellent']:
            performance_strategy = "aggressive"
        elif daily_return > self.performance_thresholds['good']:
            performance_strategy = "moderate"
        elif daily_return > self.performance_thresholds['poor']:
            performance_strategy = "conservative"
        else:
            performance_strategy = "scalping"
        
        # Market condition adjustments
        if market_volatility > self.market_thresholds['high_volatility']:
            # High volatility - be more conservative
            if performance_strategy == "aggressive":
                performance_strategy = "moderate"
            elif performance_strategy == "moderate":
                performance_strategy = "conservative"
        
        if market_trend < self.market_thresholds['bear_market']:
            # Bear market - be more conservative
            if performance_strategy in ["aggressive", "moderate"]:
                performance_strategy = "conservative"
        
        if win_rate < 0.4:
            # Low win rate - be more conservative
            if performance_strategy in ["aggressive", "moderate"]:
                performance_strategy = "conservative"
        
        # Special conditions
        if daily_return < self.performance_thresholds['terrible']:
            # Terrible performance - use scalping
            performance_strategy = "scalping"
        
        return performance_strategy
    
    def should_adjust_strategy(self, new_strategy: str) -> bool:
        """Determine if we should actually change strategies"""
        if new_strategy == self.current_strategy:
            return False
        
        # Don't change strategies too frequently
        if len(self.performance_history) > 0:
            last_change = self.performance_history[-1].get('strategy_change_time')
            if last_change:
                time_since_change = datetime.now() - datetime.fromisoformat(last_change)
                if time_since_change < timedelta(hours=6):  # Minimum 6 hours between changes
                    return False
        
        # Track strategy performance
        if self.current_strategy not in self.strategy_performance:
            self.strategy_performance[self.current_strategy] = []
        
        # Add current performance to strategy history
        current_metrics = self.calculate_performance_metrics()
        self.strategy_performance[self.current_strategy].append(current_metrics)
        
        return True
    
    async def adjust_strategy(self, new_strategy: str):
        """Adjust the trading strategy"""
        old_strategy = self.current_strategy
        self.current_strategy = new_strategy
        self.strategy = get_strategy(new_strategy)
        
        logger.info(f"🔄 Strategy Adjustment: {old_strategy} → {new_strategy}")
        logger.info(f"   Max position size: {self.strategy['max_position_size']*100}%")
        logger.info(f"   Risk per trade: {self.strategy['risk_per_trade']*100}%")
        logger.info(f"   Stop loss: {self.strategy['stop_loss']*100}%")
        logger.info(f"   Take profit: {self.strategy['take_profit']*100}%")
        
        # Save strategy change
        self.performance_history.append({
            'strategy_change_time': datetime.now().isoformat(),
            'old_strategy': old_strategy,
            'new_strategy': new_strategy,
            'reason': 'performance_based_adjustment'
        })
    
    async def analyze_asset(self, asset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a single asset with current strategy"""
        try:
            # Get market data for this asset
            market_data = await self.data_fetcher.get_market_data(asset["symbol"])
            
            if not market_data or market_data.get('current_price', 0) <= 0:
                return None
            
            # Get LLM decision for this asset
            decision = await self.llm_client.get_trading_decision(market_data)
            
            # Calculate position size based on current strategy
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
                'strategy': self.current_strategy,
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
    
    async def run_adaptive_trading_cycle(self):
        """Run one complete adaptive trading cycle"""
        try:
            logger.info("🔄 Starting adaptive trading cycle...")
            
            # 1. Assess current performance and market conditions
            performance_metrics = self.calculate_performance_metrics()
            market_conditions = await self.assess_market_conditions()
            
            # 2. Determine optimal strategy
            optimal_strategy = self.determine_optimal_strategy(performance_metrics, market_conditions)
            
            # 3. Adjust strategy if needed
            if self.should_adjust_strategy(optimal_strategy):
                await self.adjust_strategy(optimal_strategy)
            
            # 4. Log current state
            logger.info(f"📊 Performance: {performance_metrics['daily_return']*100:.2f}% daily return")
            logger.info(f"📈 Market: {market_conditions['trend']} ({market_conditions['market_trend']*100:.2f}% trend)")
            logger.info(f"⚡ Strategy: {self.current_strategy}")
            
            # 5. Analyze all assets with current strategy
            analysis_results = []
            for asset in self.assets:
                result = await self.analyze_asset(asset)
                if result:
                    analysis_results.append(result)
                    
                    # Log analysis result
                    market_data = result['market_data']
                    decision = result['decision']
                    position_calc = result['position_calculation']
                    
                    logger.info(
                        f"✅ {asset['name']} ({asset['symbol']}): "
                        f"${market_data['current_price']:.4f} | "
                        f"Decision: {decision['action']} | "
                        f"Strategy: {self.current_strategy} | "
                        f"Position: ${position_calc['actual_position_value']:.2f}"
                    )
            
            # 6. Execute trades
            for result in analysis_results:
                try:
                    asset = result['asset']
                    market_data = result['market_data']
                    decision = result['decision']
                    
                    if decision['action'] in ['BUY', 'SELL']:
                        trade_result = await self.trading_engine.execute_trade(decision, market_data)
                        logger.info(f"Trade Result for {asset['name']}: {trade_result['status']}")
                    else:
                        logger.info(f"Holding {asset['name']} - no action taken")
                        
                except Exception as e:
                    logger.error(f"Error executing trade for {asset['symbol']}: {e}")
            
            # 7. Update performance history
            self.performance_history.append(performance_metrics)
            
            # 8. Log portfolio summary
            portfolio = self.trading_engine.get_portfolio_summary()
            logger.info(f"📊 Portfolio: ${portfolio['portfolio_value']:.2f} | P&L: ${portfolio['total_pnl']:.2f} | Positions: {portfolio['position_count']}")
            
            # 9. Save analysis results
            self.save_adaptive_analysis(analysis_results, performance_metrics, market_conditions)
            
        except Exception as e:
            logger.error(f"Error in adaptive trading cycle: {e}")
    
    def save_adaptive_analysis(self, analysis_results: List[Dict[str, Any]], performance_metrics: Dict[str, Any], market_conditions: Dict[str, Any]):
        """Save adaptive analysis results"""
        try:
            filename = f"adaptive_analysis_{self.portfolio_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            data = {
                'analysis_results': analysis_results,
                'performance_metrics': performance_metrics,
                'market_conditions': market_conditions,
                'current_strategy': self.current_strategy,
                'strategy_performance': self.strategy_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Adaptive analysis results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save adaptive analysis results: {e}")
    
    async def start(self, interval_minutes: int = 15):
        """Start the adaptive multi-asset trading bot"""
        if not await self.initialize():
            logger.error("Failed to initialize adaptive bot")
            return
        
        self.is_running = True
        logger.info(f"🚀 Starting adaptive multi-asset trading bot")
        logger.info(f"📈 Portfolio: {self.portfolio_name} ({len(self.assets)} assets)")
        logger.info(f"⚡ Initial Strategy: {self.current_strategy}")
        logger.info(f"⏰ Interval: {interval_minutes} minutes")
        logger.info(f"🔄 Auto-adjustment: Enabled")
        
        try:
            while self.is_running:
                await self.run_adaptive_trading_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping bot...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the adaptive multi-asset trading bot"""
        self.is_running = False
        
        if self.llm_client:
            await self.llm_client.__aexit__(None, None, None)
        
        logger.info("Adaptive multi-asset trading bot stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        portfolio = self.trading_engine.get_portfolio_summary()
        performance_metrics = self.calculate_performance_metrics()
        
        return {
            'is_running': self.is_running,
            'portfolio_name': self.portfolio_name,
            'current_strategy': self.current_strategy,
            'assets_monitored': len(self.assets),
            'asset_list': [asset['symbol'] for asset in self.assets],
            'portfolio': portfolio,
            'current_strategy_config': self.strategy,
            'performance_metrics': performance_metrics,
            'strategy_performance_history': self.strategy_performance,
            'config': {
                'trading_enabled': config.trading_enabled,
                'paper_trading': config.paper_trading,
                'llm_model': config.llm_model
            }
        }

async def main():
    """Main function to run the adaptive multi-asset trading bot"""
    portfolio_name = "crypto_majors"  # You can change this
    
    bot = AdaptiveMultiAssetBot(portfolio_name=portfolio_name)
    
    # Run with 15 minute intervals
    await bot.start(interval_minutes=15)

if __name__ == "__main__":
    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Run the bot
    asyncio.run(main()) 