#!/usr/bin/env python3
"""
Test script for the trading bot
Run this to verify all components are working correctly
"""

import asyncio
import json
from datetime import datetime
from loguru import logger

from config import config
from llm_client import LLMClient
from data_fetcher import DataFetcher
from trading_engine import TradingEngine
from trading_bot import TradingBot

async def test_llm():
    """Test LLM connection"""
    print("🔍 Testing LLM Connection...")
    
    try:
        async with LLMClient() as llm:
            # Test basic response
            response = await llm.generate_response("What is 2+2? Answer with just the number.")
            print(f"✅ LLM Response: {response}")
            
            # Test trading decision
            market_data = {
                'symbol': 'BTC/USDT',
                'current_price': 50000,
                'price_change_24h': 2.5,
                'volume_24h': 1000000,
                'rsi': 65,
                'macd': 0.5,
                'ma_20': 49000,
                'ma_50': 48000,
                'current_position': {'side': 'None', 'size': 0},
                'portfolio_value': 10000
            }
            
            decision = await llm.get_trading_decision(market_data)
            print(f"✅ Trading Decision: {decision}")
            
            return True
            
    except Exception as e:
        print(f"❌ LLM Test Failed: {e}")
        return False

async def test_data_fetcher():
    """Test market data fetching"""
    print("\n🔍 Testing Market Data Fetcher...")
    
    try:
        fetcher = DataFetcher()
        market_data = await fetcher.get_market_data()
        
        if market_data and market_data.get('current_price', 0) > 0:
            print(f"✅ Market Data: {market_data['symbol']} @ {market_data['current_price']}")
            print(f"✅ RSI: {market_data.get('rsi', 'N/A')}")
            print(f"✅ MACD: {market_data.get('macd', 'N/A')}")
            return True
        else:
            print("❌ Invalid market data received")
            return False
            
    except Exception as e:
        print(f"❌ Data Fetcher Test Failed: {e}")
        return False

async def test_trading_engine():
    """Test trading engine"""
    print("\n🔍 Testing Trading Engine...")
    
    try:
        engine = TradingEngine()
        
        # Test paper trading
        decision = {'action': 'BUY', 'reason': 'Test trade', 'confidence': 0.8}
        market_data = {
            'symbol': 'BTC/USDT',
            'current_price': 50000,
            'portfolio_value': 10000
        }
        
        result = await engine.execute_trade(decision, market_data)
        print(f"✅ Trade Result: {result['status']}")
        
        # Test portfolio summary
        portfolio = engine.get_portfolio_summary()
        print(f"✅ Portfolio Value: ${portfolio['portfolio_value']:.2f}")
        print(f"✅ Position Count: {portfolio['position_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading Engine Test Failed: {e}")
        return False

async def test_full_bot():
    """Test the complete trading bot"""
    print("\n🔍 Testing Complete Trading Bot...")
    
    try:
        bot = TradingBot()
        
        # Test initialization
        if await bot.initialize():
            print("✅ Bot initialized successfully")
            
            # Test single cycle
            await bot.run_trading_cycle()
            print("✅ Trading cycle completed")
            
            # Test status
            status = bot.get_status()
            print(f"✅ Bot Status: Running={status['is_running']}")
            
            return True
        else:
            print("❌ Bot initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Full Bot Test Failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Trading Bot Tests...")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Test configuration
    print("🔍 Testing Configuration...")
    print(f"✅ Exchange: {config.exchange}")
    print(f"✅ Symbol: {config.symbol}")
    print(f"✅ LLM URL: {config.llm_base_url}")
    print(f"✅ LLM Model: {config.llm_model}")
    print(f"✅ Paper Trading: {config.paper_trading}")
    print(f"✅ Trading Enabled: {config.trading_enabled}")
    
    # Run tests
    tests = [
        ("LLM Client", test_llm),
        ("Data Fetcher", test_data_fetcher),
        ("Trading Engine", test_trading_engine),
        ("Full Bot", test_full_bot)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ {test_name} Test Exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your trading bot is ready to use.")
        print("\nNext steps:")
        print("1. Configure your .env file")
        print("2. Start your local LLM server")
        print("3. Run: python cli.py start")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure your local LLM server is running")
        print("2. Check your internet connection")
        print("3. Verify your configuration in .env")
    
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main()) 