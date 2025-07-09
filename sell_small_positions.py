#!/usr/bin/env python3
"""
Sell Small Positions Script
Automatically sells all cryptocurrency positions with value < €5
"""

import asyncio
from coinbase_smart_allocation_bot import CoinbaseSmartAllocationBot
from loguru import logger

async def sell_small_positions(threshold_eur: float = 5.0):
    """Sell all positions with value below threshold"""
    
    # Initialize the bot
    bot = CoinbaseSmartAllocationBot("coinbase_all_eur", "aggressive")
    
    logger.info(f"🔍 Looking for positions with value < €{threshold_eur:.2f} to sell...")
    
    # Get current positions
    existing_positions = bot.detect_existing_positions()
    
    if not existing_positions:
        logger.info("No positions found")
        return
    
    # Find positions to sell
    positions_to_sell = []
    total_value_to_free = 0
    
    for symbol, position in existing_positions.items():
        value = position['eur_value']
        if value < threshold_eur:
            positions_to_sell.append((symbol, position))
            total_value_to_free += value
            logger.info(f"📊 {symbol}: €{value:.2f} - WILL SELL")
        else:
            logger.info(f"📊 {symbol}: €{value:.2f} - keeping (≥ €{threshold_eur:.2f})")
    
    if not positions_to_sell:
        logger.info(f"✅ No positions found with value < €{threshold_eur:.2f}")
        return
    
    logger.info(f"💰 Will sell {len(positions_to_sell)} positions to free up €{total_value_to_free:.2f}")
    
    # Ask for confirmation
    print(f"\n🎯 SELLING {len(positions_to_sell)} SMALL POSITIONS:")
    for symbol, position in positions_to_sell:
        amount = position['amount']
        value = position['eur_value']
        print(f"  • {symbol}: {amount:.6f} tokens = €{value:.2f}")
    
    print(f"\n💰 Total value to be freed up: €{total_value_to_free:.2f}")
    
    confirm = input("\nProceed with selling these positions? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        logger.info("❌ Operation cancelled by user")
        return
    
    # Execute sales
    logger.info("🚀 Starting to sell positions...")
    successful_sales = 0
    
    for symbol, position in positions_to_sell:
        try:
            amount = position['amount']
            value = position['eur_value']
            
            logger.info(f"💸 Selling {symbol}: {amount:.6f} tokens = €{value:.2f}")
            
            # Place sell order
            success = bot.trading_engine.place_order(symbol, 'sell', amount, None)
            
            if success:
                logger.info(f"✅ Successfully sold {symbol}")
                successful_sales += 1
            else:
                logger.error(f"❌ Failed to sell {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error selling {symbol}: {e}")
    
    logger.info(f"🎉 Completed! Successfully sold {successful_sales}/{len(positions_to_sell)} positions")
    
    if successful_sales > 0:
        logger.info(f"💰 This should free up approximately €{total_value_to_free:.2f} for new investments")
        logger.info(f"🚀 Next bot run should trigger INVESTMENT PHASE and pick ONE best crypto!")

async def main():
    """Main function"""
    await sell_small_positions(threshold_eur=5.0)

if __name__ == "__main__":
    # Run the script
    asyncio.run(main()) 