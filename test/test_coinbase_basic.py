#!/usr/bin/env python3
"""
Test Basic Coinbase API Connection
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

def main():
    print("🔗 Testing Basic Coinbase API Connection...")
    
    # Get API credentials from environment
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in .env file.")
        return
    
    try:
        # Initialize Coinbase exchange
        exchange = ccxt.coinbase({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': False,  # Set to True for sandbox testing
            'enableRateLimit': True
        })
        
        print("✅ Connected to Coinbase API!")
        
        # Test basic API calls
        print("\n📊 Testing market data...")
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"BTC/USDT: ${ticker['last']:.2f}")
        
        print("\n👤 Testing account info...")
        balance = exchange.fetch_balance()
        print(f"Total balance: {len(balance['total'])} currencies")
        
        # Show some balances
        for currency, amount in balance['total'].items():
            if amount > 0:
                print(f"  {currency}: {amount}")
        
        print("\n✅ All basic tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 