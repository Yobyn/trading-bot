#!/usr/bin/env python3
"""
Test Coinbase Advanced API Connection (Private key from file)
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Add parent directory to path and load .env from there
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
load_dotenv(parent_dir / '.env')

api_key = os.getenv('COINBASE_API_KEY')
api_secret_file = os.getenv('COINBASE_API_SECRET_FILE')

# If api_secret_file is relative, make it relative to parent directory
if api_secret_file and not os.path.isabs(api_secret_file):
    api_secret_file = str(parent_dir / api_secret_file)


def main():
    print("🔗 Testing Coinbase Advanced API Connection...")
    if not api_key or not api_secret_file:
        print("❌ API key or secret file not found in .env file.")
        return
    try:
        with open(api_secret_file, 'r') as f:
            api_secret = f.read()
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        print("✅ Connected to Coinbase Advanced API!")
        
        print("\n👤 Fetching account info...")
        accounts = client.get_accounts()
        print(f"✅ Accounts fetched: {len(accounts.accounts)}")
        
        # Inspect account structure
        if accounts.accounts:
            account = accounts.accounts[0]
            print(f"\n📋 Account structure:")
            print(f"Type: {type(account)}")
            print(f"Dir: {dir(account)}")
            if hasattr(account, 'available_balance'):
                balance = account.available_balance
                print(f"Balance type: {type(balance)}")
                print(f"Balance dir: {dir(balance)}")
                if hasattr(balance, 'value'):
                    print(f"Value: {balance.value}")
                if hasattr(balance, 'currency'):
                    print(f"Currency: {balance.currency}")
        
        print("\n📊 Testing market data...")
        ticker = client.get_best_bid_ask(product_id="BTC-USD")
        print(f"Ticker type: {type(ticker)}")
        print(f"Ticker dir: {dir(ticker)}")
        if hasattr(ticker, 'pricebooks'):
            print(f"Pricebooks: {ticker.pricebooks}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 