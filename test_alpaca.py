#!/usr/bin/env python3
"""
Test script for Alpaca integration
"""

import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from alpaca_fetcher import AlpacaDataFetcher

# Load environment variables
load_dotenv()

async def test_alpaca():
    """Test Alpaca connection and data fetching"""
    print("🔍 Testing Alpaca Connection...")
    
    try:
        # Get API keys from environment variables
        api_key = os.getenv('EXCHANGE_API_KEY')
        secret_key = os.getenv('EXCHANGE_SECRET')
        
        if not api_key or not secret_key:
            print("❌ API keys not found in .env file")
            print("💡 Add your Alpaca API keys to .env file:")
            print("   EXCHANGE_API_KEY=your_api_key_here")
            print("   EXCHANGE_SECRET=your_secret_key_here")
            return False
        
        fetcher = AlpacaDataFetcher(api_key=api_key, secret_key=secret_key, paper=True)
        
        # Test market data
        market_data = await fetcher.get_market_data('AAPL', '1h')
        
        if market_data and market_data.get('current_price', 0) > 0:
            print(f"✅ Market Data: {market_data['symbol']} @ ${market_data['current_price']:.2f}")
            print(f"✅ RSI: {market_data.get('rsi', 'N/A'):.2f}")
            print(f"✅ MACD: {market_data.get('macd', 'N/A'):.2f}")
            print(f"✅ 24h Change: {market_data.get('price_change_24h', 'N/A'):.2f}%")
            print(f"✅ Volume: {market_data.get('volume_24h', 'N/A'):,.0f}")
            return True
        else:
            print("❌ Invalid market data received")
            return False
            
    except Exception as e:
        print(f"❌ Alpaca Test Failed: {e}")
        return False

async def main():
    """Run Alpaca test"""
    print("🚀 Testing Alpaca Integration...")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    success = await test_alpaca()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Alpaca integration working! Ready for paper trading.")
        print("\nNext steps:")
        print("1. Run: python cli.py start")
        print("2. Your bot will trade AAPL stock with AI decisions")
        print("3. All trades will be paper trading (no real money)")
    else:
        print("⚠️ Alpaca test failed. Check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main()) 