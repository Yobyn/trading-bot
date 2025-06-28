#!/usr/bin/env python3
"""
Coinbase Smart Allocation Bot CLI
Command-line interface for the Coinbase smart allocation bot
"""

import asyncio
import argparse
from coinbase_smart_allocation_bot import CoinbaseSmartAllocationBot
from config import config

async def main():
    parser = argparse.ArgumentParser(description='Coinbase Smart Allocation Bot CLI')
    
    parser.add_argument('--portfolio', '-p', 
                       choices=['coinbase_majors', 'coinbase_all_eur', 'coinbase_majors_usd'],
                       default='coinbase_majors',
                       help='Portfolio to trade (default: coinbase_majors for EUR, coinbase_all_eur for all 41 cryptos)')
    
    parser.add_argument('--strategy', '-s',
                       choices=['conservative', 'moderate', 'aggressive', 'scalping'],
                       default='moderate',
                       help='Trading strategy (default: moderate)')
    
    parser.add_argument('--interval', '-i',
                       type=int,
                       default=15,
                       help='Trading interval in minutes (default: 15)')
    
    parser.add_argument('--test', '-t',
                       action='store_true',
                       help='Run a single test cycle and exit')
    
    parser.add_argument('--paper-balance', '-b',
                       type=float,
                       default=10000.0,
                       help='Paper trading balance in EUR (default: 10000.0)')
    
    args = parser.parse_args()
    
    # Create bot
    bot = CoinbaseSmartAllocationBot(
        portfolio_name=args.portfolio,
        strategy=args.strategy
    )
    
    print("🚀 Coinbase Smart Allocation Bot CLI")
    print("=" * 50)
    print(f"📈 Portfolio: {args.portfolio}")
    print(f"⚡ Strategy: {args.strategy}")
    print(f"⏰ Interval: {args.interval} minutes")
    print(f"🧪 Test Mode: {'Yes' if args.test else 'No'}")
    
    # Fetch live balance if live trading is enabled (after showing basic info)
    if config.trading_enabled:
        # Initialize bot first to ensure proper setup
        await bot.initialize()
        # Get the corrected account balance that includes crypto holdings
        live_balance = bot.trading_engine.get_account_balance()
        print(f"💰 Coinbase Balance: €{live_balance:.2f}")
    else:
        print(f"💸 Paper Balance: {args.paper_balance} EUR")
    
    print("=" * 50)
    
    # Set paper trading balance if in test mode and paper trading is enabled
    if args.test and not config.trading_enabled:
        bot.trading_engine.set_paper_balance(args.paper_balance)
    
    if args.test:
        # Run single test cycle
        print("\n🧪 Running test cycle...")
        await bot.initialize()
        await bot.run_smart_allocation_cycle()
        print("✅ Test cycle completed!")
    else:
        # Start continuous bot
        print("\n🚀 Starting Coinbase Smart Allocation Bot...")
        while True:
            try:
                await bot.run_smart_allocation_cycle()
                print(f"⏰ Waiting {args.interval} minutes before next cycle...")
                await asyncio.sleep(args.interval * 60)
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user (KeyboardInterrupt)") 