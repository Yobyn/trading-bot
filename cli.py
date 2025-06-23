#!/usr/bin/env python3
import asyncio
import argparse
import json
from datetime import datetime
from trading_bot import TradingBot
from loguru import logger

async def start_bot(args):
    """Start the trading bot"""
    bot = TradingBot()
    
    try:
        await bot.start(interval_minutes=args.interval)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")

async def status_bot(args):
    """Show bot status"""
    bot = TradingBot()
    
    if await bot.initialize():
        status = bot.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Failed to initialize bot")

async def test_llm(args):
    """Test LLM connection"""
    from llm_client import LLMClient
    
    async with LLMClient() as llm:
        test_prompt = "What is 2+2? Answer with just the number."
        response = await llm.generate_response(test_prompt)
        print(f"LLM Response: {response}")

async def test_market_data(args):
    """Test market data fetching"""
    from data_fetcher import DataFetcher
    
    fetcher = DataFetcher()
    market_data = await fetcher.get_market_data()
    print(json.dumps(market_data, indent=2, default=str))

async def run_single_cycle(args):
    """Run a single trading cycle"""
    bot = TradingBot()
    
    if await bot.initialize():
        await bot.run_trading_cycle()
        status = bot.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Failed to initialize bot")

def main():
    parser = argparse.ArgumentParser(description="Trading Bot CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the trading bot')
    start_parser.add_argument('--interval', type=int, default=60, 
                             help='Trading cycle interval in minutes (default: 60)')
    
    # Status command
    subparsers.add_parser('status', help='Show bot status')
    
    # Test commands
    subparsers.add_parser('test-llm', help='Test LLM connection')
    subparsers.add_parser('test-data', help='Test market data fetching')
    
    # Single cycle command
    subparsers.add_parser('cycle', help='Run a single trading cycle')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        asyncio.run(start_bot(args))
    elif args.command == 'status':
        asyncio.run(status_bot(args))
    elif args.command == 'test-llm':
        asyncio.run(test_llm(args))
    elif args.command == 'test-data':
        asyncio.run(test_market_data(args))
    elif args.command == 'cycle':
        asyncio.run(run_single_cycle(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 