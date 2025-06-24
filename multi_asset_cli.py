#!/usr/bin/env python3
"""
Multi-Asset Trading Bot CLI
"""

import asyncio
import argparse
import json
from datetime import datetime
from multi_asset_bot import MultiAssetTradingBot
from loguru import logger

async def start_multi_bot(args):
    """Start the multi-asset trading bot"""
    bot = MultiAssetTradingBot()
    
    try:
        await bot.start(interval_minutes=args.interval)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")

async def test_multi_bot(args):
    """Test the multi-asset bot with a single cycle"""
    bot = MultiAssetTradingBot()
    
    if await bot.initialize():
        await bot.run_trading_cycle()
        status = bot.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Failed to initialize multi-asset bot")

async def status_multi_bot(args):
    """Show multi-asset bot status"""
    bot = MultiAssetTradingBot()
    
    if await bot.initialize():
        status = bot.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Failed to initialize multi-asset bot")

def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Trading Bot CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the multi-asset trading bot')
    start_parser.add_argument('--interval', type=int, default=15, 
                             help='Trading cycle interval in minutes (default: 15)')
    
    # Test command
    subparsers.add_parser('test', help='Test multi-asset bot with single cycle')
    
    # Status command
    subparsers.add_parser('status', help='Show multi-asset bot status')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        asyncio.run(start_multi_bot(args))
    elif args.command == 'test':
        asyncio.run(test_multi_bot(args))
    elif args.command == 'status':
        asyncio.run(status_multi_bot(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 