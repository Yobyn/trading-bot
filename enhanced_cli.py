#!/usr/bin/env python3
"""
Enhanced Multi-Asset Trading Bot CLI
"""

import asyncio
import argparse
import json
from datetime import datetime
from enhanced_multi_bot import EnhancedMultiAssetBot
from asset_config import list_available_portfolios, list_available_strategies
from loguru import logger

async def start_enhanced_bot(args):
    """Start the enhanced multi-asset trading bot"""
    bot = EnhancedMultiAssetBot(
        portfolio_name='coinbase_majors',
        strategy_name=args.strategy
    )
    
    try:
        await bot.start(interval_minutes=args.interval)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")

async def test_enhanced_bot(args):
    """Test the enhanced multi-asset bot with a single cycle"""
    bot = EnhancedMultiAssetBot(
        portfolio_name='coinbase_majors',
        strategy_name=args.strategy
    )
    
    if await bot.initialize():
        await bot.run_trading_cycle()
        status = bot.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Failed to initialize enhanced multi-asset bot")

async def status_enhanced_bot(args):
    """Show enhanced multi-asset bot status"""
    bot = EnhancedMultiAssetBot(
        portfolio_name='coinbase_majors',
        strategy_name=args.strategy
    )
    
    if await bot.initialize():
        status = bot.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Failed to initialize enhanced multi-asset bot")

def list_options(args):
    """List available portfolios and strategies"""
    print("Available Portfolios:")
    for portfolio in list_available_portfolios():
        print(f"  - {portfolio}")
    
    print("\nAvailable Strategies:")
    for strategy in list_available_strategies():
        print(f"  - {strategy}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-Asset Trading Bot CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the enhanced multi-asset trading bot')
    start_parser.add_argument('--portfolio', type=str, default='coinbase_majors',
                             help='Portfolio name (default: coinbase_majors)')
    start_parser.add_argument('--strategy', type=str, default='moderate',
                             help='Trading strategy (default: moderate)')
    start_parser.add_argument('--interval', type=int, default=15,
                             help='Trading cycle interval in minutes (default: 15)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test enhanced multi-asset bot with single cycle')
    test_parser.add_argument('--portfolio', type=str, default='coinbase_majors',
                            help='Portfolio name (default: coinbase_majors)')
    test_parser.add_argument('--strategy', type=str, default='moderate',
                            help='Trading strategy (default: moderate)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show enhanced multi-asset bot status')
    status_parser.add_argument('--portfolio', type=str, default='coinbase_majors',
                              help='Portfolio name (default: coinbase_majors)')
    status_parser.add_argument('--strategy', type=str, default='moderate',
                              help='Trading strategy (default: moderate)')
    
    # List command
    subparsers.add_parser('list', help='List available portfolios and strategies')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        asyncio.run(start_enhanced_bot(args))
    elif args.command == 'test':
        asyncio.run(test_enhanced_bot(args))
    elif args.command == 'status':
        asyncio.run(status_enhanced_bot(args))
    elif args.command == 'list':
        list_options(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 