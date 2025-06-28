#!/usr/bin/env python3
"""
Test Coinbase Exchanges
Check which Coinbase exchange provides the best data access and trading capabilities
"""

import asyncio
import ccxt
from datetime import datetime
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path and load .env from there
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
load_dotenv(parent_dir / '.env')

async def test_coinbase_exchanges():
    """Test different Coinbase exchanges"""
    print("🔍 Testing Coinbase Exchanges")
    print("=" * 50)
    
    # Test different Coinbase exchanges
    coinbase_exchanges = [
        "coinbase",
        "coinbaseadvanced", 
        "coinbaseexchange",
        "coinbaseinternational"
    ]
    
    # Test symbols
    test_symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"]
    
    results = {}
    
    for exchange_name in coinbase_exchanges:
        print(f"\n🔍 Testing {exchange_name.upper()}...")
        
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'sandbox': False,  # Coinbase doesn't have sandbox
            })
            
            # Load markets
            markets = exchange.load_markets()
            print(f"  ✅ Connected successfully")
            print(f"  📊 Markets loaded: {len(markets)}")
            
            # Test data access
            data_results = {}
            for symbol in test_symbols:
                try:
                    # Check if symbol exists
                    if symbol in markets:
                        # Test ticker
                        ticker = exchange.fetch_ticker(symbol)
                        if ticker and ticker.get('last'):
                            data_results[symbol] = {
                                'price': ticker['last'],
                                'volume_24h': ticker.get('quoteVolume', 0),
                                'change_24h': ticker.get('percentage', 0)
                            }
                            print(f"    ✅ {symbol}: ${ticker['last']:.2f}")
                        else:
                            print(f"    ❌ {symbol}: No data")
                    else:
                        print(f"    ❌ {symbol}: Not available")
                        
                except Exception as e:
                    print(f"    ❌ {symbol}: Error - {str(e)[:50]}...")
            
            results[exchange_name] = {
                'connected': True,
                'markets_count': len(markets),
                'data_access': data_results,
                'available_symbols': len(data_results)
            }
            
        except Exception as e:
            print(f"  ❌ Connection failed: {str(e)[:50]}...")
            results[exchange_name] = {
                'connected': False,
                'error': str(e)
            }
    
    # Summary
    print(f"\n📊 COINBASE TESTING SUMMARY")
    print("=" * 50)
    
    successful_exchanges = []
    for exchange_name, result in results.items():
        if result.get('connected'):
            available_symbols = result.get('available_symbols', 0)
            markets_count = result.get('markets_count', 0)
            print(f"✅ {exchange_name.upper()}: {available_symbols}/{len(test_symbols)} symbols, {markets_count} total markets")
            successful_exchanges.append({
                'name': exchange_name,
                'available_symbols': available_symbols,
                'markets_count': markets_count,
                'data_access': result.get('data_access', {})
            })
        else:
            print(f"❌ {exchange_name.upper()}: Failed to connect")
    
    # Recommendations
    print(f"\n🎯 COINBASE RECOMMENDATIONS")
    print("=" * 50)
    
    if successful_exchanges:
        # Sort by available symbols
        successful_exchanges.sort(key=lambda x: x['available_symbols'], reverse=True)
        
        print(f"🏆 Best Coinbase Exchange for Your Trading Bot:")
        for i, exchange in enumerate(successful_exchanges[:3], 1):
            print(f"  {i}. {exchange['name'].upper()}")
            print(f"     • Available symbols: {exchange['available_symbols']}/{len(test_symbols)}")
            print(f"     • Total markets: {exchange['markets_count']}")
            
            # Show sample data
            if exchange['data_access']:
                sample_symbol = list(exchange['data_access'].keys())[0]
                sample_data = exchange['data_access'][sample_symbol]
                print(f"     • Sample: {sample_symbol} @ ${sample_data['price']:.2f}")
            print()
    
    # API setup instructions
    print(f"🔧 COINBASE API SETUP")
    print("=" * 50)
    print(f"1. Go to https://www.coinbase.com/settings/api")
    print(f"2. Create a new API key")
    print(f"3. Enable permissions for:")
    print(f"   • View (for data access)")
    print(f"   • Trade (for trading)")
    print(f"4. Add to your .env file:")
    print(f"   COINBASE_API_KEY=your_api_key")
    print(f"   COINBASE_SECRET=your_secret_key")
    print(f"   COINBASE_PASSPHRASE=your_passphrase")
    print()
    
    return results

def test_legacy_coinbase():
    """Test legacy Coinbase API using ccxt"""
    print("🔗 Testing Legacy Coinbase API (ccxt)...")
    
    # Get API credentials
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in .env file")
        return False
    
    try:
        # Initialize legacy Coinbase exchange
        exchange = ccxt.coinbase({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        
        print("✅ Connected to Legacy Coinbase API!")
        
        # Test market data
        print("\n📊 Testing market data...")
        markets = exchange.load_markets()
        print(f"Available markets: {len(markets)}")
        
        # Test specific symbols
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        for symbol in test_symbols:
            if symbol in markets:
                ticker = exchange.fetch_ticker(symbol)
                print(f"✅ {symbol}: ${ticker['last']:.2f}")
            else:
                print(f"⚠️  {symbol} not available")
        
        # Test account balance
        print("\n👤 Testing account balance...")
        balance = exchange.fetch_balance()
        print(f"Account has {len(balance['total'])} currencies")
        
        # Show non-zero balances
        for currency, amount in balance['total'].items():
            if amount > 0:
                print(f"  {currency}: {amount}")
        
        # Test order book
        print("\n📚 Testing order book...")
        orderbook = exchange.fetch_order_book('BTC/USDT', limit=5)
        print(f"BTC/USDT order book - Best bid: ${orderbook['bids'][0][0]:.2f}")
        
        print("\n✅ Legacy Coinbase API test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_public_endpoints():
    """Test public endpoints without API credentials"""
    print("\n🌐 Testing Public Endpoints...")
    
    try:
        # Initialize without credentials
        exchange = ccxt.coinbase({
            'enableRateLimit': True
        })
        
        # Test public market data
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✅ Public ticker accessible - BTC/USDT: ${ticker['last']:.2f}")
        
        # Test order book
        orderbook = exchange.fetch_order_book('BTC/USDT', limit=5)
        print(f"✅ Public order book accessible")
        
        # Test recent trades
        trades = exchange.fetch_trades('BTC/USDT', limit=5)
        print(f"✅ Public trades accessible - {len(trades)} trades")
        
        print("✅ Public endpoints test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Public endpoints error: {e}")
        return False

def main():
    """Run all legacy Coinbase tests"""
    print("🚀 Starting Legacy Coinbase API Tests...\n")
    
    # Test public endpoints
    public_ok = test_public_endpoints()
    
    # Test authenticated endpoints
    auth_ok = test_legacy_coinbase()
    
    # Summary
    print("\n" + "="*50)
    print("📋 LEGACY API TEST SUMMARY")
    print("="*50)
    print(f"Public Endpoints: {'✅ PASS' if public_ok else '❌ FAIL'}")
    print(f"Authenticated API: {'✅ PASS' if auth_ok else '❌ FAIL'}")
    
    if public_ok and auth_ok:
        print("\n🎉 All legacy API tests passed!")
    else:
        print("\n⚠️  Some legacy API tests failed.")

if __name__ == "__main__":
    main() 