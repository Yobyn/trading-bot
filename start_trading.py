#!/usr/bin/env python3
"""
Simple Multi-Asset Trading Bot Starter
"""

import asyncio
import sys
from enhanced_multi_bot import EnhancedMultiAssetBot
from asset_config import list_available_portfolios, list_available_strategies

def print_banner():
    print("🚀 Multi-Asset Trading Bot")
    print("=" * 40)

def print_portfolios():
    print("\n📈 Available Portfolios:")
    portfolios = list_available_portfolios()
    for i, portfolio in enumerate(portfolios, 1):
        print(f"  {i}. {portfolio}")
    return portfolios

def print_strategies():
    print("\n⚡ Available Strategies:")
    strategies = list_available_strategies()
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy}")
    return strategies

def get_user_choice(options, prompt):
    while True:
        try:
            choice = input(f"\n{prompt} (1-{len(options)}): ").strip()
            if choice.lower() == 'q':
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return options[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def get_interval():
    while True:
        try:
            interval = input("\n⏰ Trading interval in minutes (default: 15): ").strip()
            if not interval:
                return 15
            interval_num = int(interval)
            if interval_num > 0:
                return interval_num
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")

async def main():
    print_banner()
    
    # Show available portfolios
    portfolios = print_portfolios()
    portfolio = get_user_choice(portfolios, "Select a portfolio")
    if not portfolio:
        print("Goodbye!")
        return
    
    # Show available strategies
    strategies = print_strategies()
    strategy = get_user_choice(strategies, "Select a trading strategy")
    if not strategy:
        print("Goodbye!")
        return
    
    # Get trading interval
    interval = get_interval()
    
    # Confirm settings
    print(f"\n🎯 Trading Configuration:")
    print(f"   Portfolio: {portfolio}")
    print(f"   Strategy: {strategy}")
    print(f"   Interval: {interval} minutes")
    
    confirm = input("\nStart trading? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Trading cancelled.")
        return
    
    # Start the bot
    print(f"\n🚀 Starting {portfolio} portfolio with {strategy} strategy...")
    print("Press Ctrl+C to stop the bot")
    
    bot = EnhancedMultiAssetBot(portfolio_name='coinbase_majors', strategy_name=strategy)
    
    try:
        await bot.start(interval_minutes=interval)
    except KeyboardInterrupt:
        print("\n\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!") 