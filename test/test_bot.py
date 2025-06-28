#!/usr/bin/env python3
"""
Test General Bot Functionality
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path and load .env from there
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
load_dotenv(parent_dir / '.env')

def test_environment():
    """Test environment configuration"""
    print("🔧 Testing Environment Configuration...")
    
    # Check required environment variables
    required_vars = [
        'COINBASE_API_KEY',
        'COINBASE_API_SECRET_FILE'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("✅ All required environment variables found")
        return True

def test_config_files():
    """Test configuration files"""
    print("\n📁 Testing Configuration Files...")
    
    # Check if key files exist
    required_files = [
        'config.py',
        'asset_config.py',
        'llm_client.py',
        'data_fetcher.py',
        'trading_engine.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing configuration files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All configuration files found")
        return True

def test_imports():
    """Test importing main modules"""
    print("\n📦 Testing Module Imports...")
    
    try:
        # Test importing main modules
        import config
        print("✅ config module imported")
        
        import asset_config
        print("✅ asset_config module imported")
        
        import llm_client
        print("✅ llm_client module imported")
        
        import data_fetcher
        print("✅ data_fetcher module imported")
        
        import trading_engine
        print("✅ trading_engine module imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_portfolio_config():
    """Test portfolio configuration"""
    print("\n💼 Testing Portfolio Configuration...")
    
    try:
        import asset_config
        
        # Test getting portfolios
        portfolios = asset_config.list_available_portfolios()
        print(f"✅ Available portfolios: {portfolios}")
        
        # Test getting a specific portfolio
        portfolio = asset_config.get_portfolio('coinbase_majors')
        if portfolio:
            print(f"✅ Portfolio 'coinbase_majors' loaded with {len(portfolio)} assets")
        else:
            print("❌ Failed to load portfolio 'coinbase_majors'")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio configuration error: {e}")
        return False

def test_strategy_config():
    """Test strategy configuration"""
    print("\n🎯 Testing Strategy Configuration...")
    
    try:
        import asset_config
        
        # Test getting strategies
        strategies = asset_config.list_available_strategies()
        print(f"✅ Available strategies: {strategies}")
        
        # Test getting a specific strategy
        strategy = asset_config.get_strategy('moderate')
        if strategy:
            print(f"✅ Strategy 'moderate' loaded")
            print(f"   Max position size: {strategy.get('max_position_size', 'N/A')}")
            print(f"   Risk tolerance: {strategy.get('risk_tolerance', 'N/A')}")
        else:
            print("❌ Failed to load strategy 'moderate'")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy configuration error: {e}")
        return False

def main():
    """Run all bot functionality tests"""
    print("🚀 Starting Bot Functionality Tests...\n")
    
    # Run all tests
    tests = [
        ("Environment", test_environment),
        ("Config Files", test_config_files),
        ("Module Imports", test_imports),
        ("Portfolio Config", test_portfolio_config),
        ("Strategy Config", test_strategy_config)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📋 BOT FUNCTIONALITY TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All bot functionality tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check your configuration.")

if __name__ == "__main__":
    main() 