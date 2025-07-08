#!/usr/bin/env python3
"""
Cleanup Position History Script
Removes entries from position_history.json for positions that no longer exist in Coinbase
"""

import asyncio
from coinbase_smart_allocation_bot import CoinbaseSmartAllocationBot
from loguru import logger

async def main():
    """Main cleanup function"""
    logger.info("🧹 Position History Cleanup Tool")
    logger.info("=" * 50)
    
    # Initialize the bot (we only need it for the cleanup functionality)
    bot = CoinbaseSmartAllocationBot()
    
    # Run the cleanup
    bot.cleanup_position_history()
    
    logger.info("=" * 50)
    logger.info("✅ Cleanup complete!")

if __name__ == "__main__":
    asyncio.run(main()) 