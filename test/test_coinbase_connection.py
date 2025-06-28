#!/usr/bin/env python3
"""
Test Coinbase Connection and Authentication
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import ccxt

# Add parent directory to path and load .env from there
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
load_dotenv(parent_dir / '.env')

def test_connection():
    """Test basic connection to Coinbase"""
    print("🔗 Testing Coinbase Connection...")
    
    try:
        # Initialize without API credentials for basic connection test
        exchange = ccxt.coinbase({
            'enableRateLimit': True
        })
        
        # Test basic connectivity
        markets = exchange.load_markets()
        print(f"✅ Connected to Coinbase! Available markets: {len(markets)}")
        
        # Test a simple API call
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✅ Market data accessible - BTC/USDT: ${ticker['last']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_authentication():
    """Test API authentication"""
    print("\n🔐 Testing API Authentication...")
    
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in .env file")
        return False
    
    try:
        # Initialize with API credentials
        exchange = ccxt.coinbase({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        
        # Test authenticated API call
        balance = exchange.fetch_balance()
        print(f"✅ Authentication successful! Account has {len(balance['total'])} currencies")
        
        # Show some account info
        total_balance = sum(amount for amount in balance['total'].values() if amount > 0)
        print(f"✅ Total balance across all currencies: {total_balance}")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def test_trading_capabilities():
    """Test trading-related API calls"""
    print("\n💰 Testing Trading Capabilities...")
    
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ API credentials required for trading tests")
        return False
    
    try:
        exchange = ccxt.coinbase({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        
        # Test order book access
        orderbook = exchange.fetch_order_book('BTC/USDT', limit=5)
        print(f"✅ Order book accessible - Best bid: ${orderbook['bids'][0][0]:.2f}")
        
        # Test recent trades
        trades = exchange.fetch_trades('BTC/USDT', limit=5)
        print(f"✅ Recent trades accessible - {len(trades)} trades fetched")
        
        # Test account orders (if any)
        orders = exchange.fetch_open_orders('BTC/USDT')
        print(f"✅ Open orders accessible - {len(orders)} open orders")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading test failed: {e}")
        return False

def main():
    """Run all connection tests"""
    print("🚀 Starting Coinbase Connection Tests...\n")
    
    # Test basic connection
    connection_ok = test_connection()
    
    # Test authentication
    auth_ok = test_authentication()
    
    # Test trading capabilities
    trading_ok = test_trading_capabilities()
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    print(f"Connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    print(f"Authentication: {'✅ PASS' if auth_ok else '❌ FAIL'}")
    print(f"Trading: {'✅ PASS' if trading_ok else '❌ FAIL'}")
    
    if connection_ok and auth_ok and trading_ok:
        print("\n🎉 All tests passed! Coinbase integration is ready.")
    else:
        print("\n⚠️  Some tests failed. Check your configuration.")

if __name__ == "__main__":
    main() 