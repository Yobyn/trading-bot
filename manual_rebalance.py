#!/usr/bin/env python3
"""
Manual Rebalancing Script
Triggers a complete portfolio rebalancing cycle
"""

import asyncio
from datetime import datetime
from coinbase_smart_allocation_bot import CoinbaseSmartAllocationBot
from loguru import logger
from datetime import datetime

async def manual_rebalance():
    """Manually trigger a complete portfolio rebalancing"""
    
    print("🔄 Manual Portfolio Rebalancing")
    print("=" * 50)
    
    # Ask for confirmation
    response = input("⚠️  This will LIQUIDATE ALL current positions and reinvest based on LLM recommendations.\nAre you sure? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("❌ Rebalancing cancelled.")
        return
    
    # Initialize the bot
    portfolio_name = "coinbase_all_eur"  # Change if you use a different portfolio
    strategy_name = "aggressive"         # Change to match your strategy
    
    bot = CoinbaseSmartAllocationBot(portfolio_name, strategy_name)
    
    try:
        # Show current positions before rebalancing
        print("\n📊 Current Portfolio:")
        existing_positions = bot.detect_existing_positions()
        if existing_positions:
            total_value = 0
            for symbol, position in existing_positions.items():
                print(f"  {symbol}: {position['amount']:.4f} = €{position['eur_value']:.2f}")
                total_value += position['eur_value']
            print(f"  Total Position Value: €{total_value:.2f}")
        else:
            print("  No positions found")
        
        print(f"\n💰 Available Cash: €{bot.get_eur_balance():.2f}")
        print(f"📊 Total Portfolio: €{bot.trading_engine.get_account_balance():.2f}")
        
        # Final confirmation
        print("\n🚨 FINAL CONFIRMATION")
        final_response = input("Type 'REBALANCE' to proceed: ")
        
        if final_response != 'REBALANCE':
            print("❌ Rebalancing cancelled.")
            return
        
        # Execute rebalancing
        print("\n🔄 Starting rebalancing cycle...")
        should_continue = await bot.run_rebalancing_cycle()
        
        if should_continue:
            print("✅ Rebalancing completed successfully!")
            
            # Show new positions
            print("\n📊 New Portfolio:")
            new_positions = bot.detect_existing_positions()
            if new_positions:
                total_value = 0
                for symbol, position in new_positions.items():
                    print(f"  {symbol}: {position['amount']:.4f} = €{position['eur_value']:.2f}")
                    total_value += position['eur_value']
                print(f"  Total Position Value: €{total_value:.2f}")
            
            print(f"\n💰 Remaining Cash: €{bot.get_eur_balance():.2f}")
            print(f"📊 Total Portfolio: €{bot.trading_engine.get_account_balance():.2f}")
        else:
            print("❌ Rebalancing failed or was incomplete.")
            
    except Exception as e:
        logger.error(f"Error during rebalancing: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🤖 Manual Portfolio Rebalancing Tool")
    print("📅 Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    asyncio.run(manual_rebalance()) 