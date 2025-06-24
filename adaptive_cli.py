#!/usr/bin/env python3
"""
Adaptive Multi-Asset Trading Bot CLI
"""

import asyncio
import argparse
import json
from datetime import datetime
from adaptive_bot import AdaptiveMultiAssetBot
from asset_config import list_available_portfolios
from loguru import logger

async def start_adaptive_bot(args):
    """Start the adaptive multi-asset trading bot"""
    bot = AdaptiveMultiAssetBot(portfolio_name=args.portfolio)
    
    try:
        await bot.start(interval_minutes=args.interval)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")

async def test_adaptive_bot(args):
    """Test the adaptive multi-asset bot with a single cycle"""
    bot = AdaptiveMultiAssetBot(portfolio_name=args.portfolio)
    
    if await bot.initialize():
        await bot.run_adaptive_trading_cycle()
        status = bot.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Failed to initialize adaptive multi-asset bot")

async def status_adaptive_bot(args):
    """Show adaptive multi-asset bot status"""
    bot = AdaptiveMultiAssetBot(portfolio_name=args.portfolio)
    
    if await bot.initialize():
        status = bot.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Failed to initialize adaptive multi-asset bot")

def print_adaptive_features():
    """Print information about adaptive features"""
    print("🤖 Adaptive Trading Bot Features:")
    print("=" * 50)
    print("📊 Performance-Based Adjustments:")
    print("  • Excellent (>5% daily): Switch to Aggressive")
    print("  • Good (2-5% daily): Stay Moderate")
    print("  • Poor (-2% to 2% daily): Switch to Conservative")
    print("  • Terrible (<-2% daily): Switch to Scalping")
    print()
    print("📈 Market Condition Adjustments:")
    print("  • High Volatility: Reduce risk (Aggressive → Moderate)")
    print("  • Bear Market: Increase conservatism")
    print("  • Low Win Rate (<40%): Switch to Conservative")
    print()
    print("⚡ Strategy Parameters:")
    print("  • Conservative: 5% max position, 10% stop loss")
    print("  • Moderate: 10% max position, 15% stop loss")
    print("  • Aggressive: 20% max position, 25% stop loss")
    print("  • Scalping: 5% max position, 5% stop loss")
    print()
    print("🔄 Auto-Adjustment Rules:")
    print("  • Minimum 6 hours between strategy changes")
    print("  • Tracks performance history for each strategy")
    print("  • Considers market volatility and trends")
    print("  • Adjusts based on win rate and daily returns")

def main():
    parser = argparse.ArgumentParser(description="Adaptive Multi-Asset Trading Bot CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the adaptive multi-asset trading bot')
    start_parser.add_argument('--portfolio', type=str, default='crypto_majors',
                             help='Portfolio name (default: crypto_majors)')
    start_parser.add_argument('--interval', type=int, default=15,
                             help='Trading cycle interval in minutes (default: 15)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test adaptive multi-asset bot with single cycle')
    test_parser.add_argument('--portfolio', type=str, default='crypto_majors',
                            help='Portfolio name (default: crypto_majors)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show adaptive multi-asset bot status')
    status_parser.add_argument('--portfolio', type=str, default='crypto_majors',
                              help='Portfolio name (default: crypto_majors)')
    
    # Features command
    subparsers.add_parser('features', help='Show adaptive bot features')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        asyncio.run(start_adaptive_bot(args))
    elif args.command == 'test':
        asyncio.run(test_adaptive_bot(args))
    elif args.command == 'status':
        asyncio.run(status_adaptive_bot(args))
    elif args.command == 'features':
        print_adaptive_features()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 