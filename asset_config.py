#!/usr/bin/env python3
"""
Asset Configuration for Multi-Asset Trading Bot (Coinbase Only)
"""

# Predefined asset portfolios
ASSET_PORTFOLIOS = {
    "coinbase_majors": [
        {"symbol": "BTC/EUR", "name": "Bitcoin", "exchange": "coinbase"},
        {"symbol": "ETH/EUR", "name": "Ethereum", "exchange": "coinbase"},
        {"symbol": "SOL/EUR", "name": "Solana", "exchange": "coinbase"},
        {"symbol": "ADA/EUR", "name": "Cardano", "exchange": "coinbase"},
    ],
    "coinbase_all_eur": [
        # Major Cryptocurrencies
        {"symbol": "BTC/EUR", "name": "Bitcoin", "exchange": "coinbase"},
        {"symbol": "ETH/EUR", "name": "Ethereum", "exchange": "coinbase"},
        {"symbol": "LTC/EUR", "name": "Litecoin", "exchange": "coinbase"},
        {"symbol": "BCH/EUR", "name": "Bitcoin Cash", "exchange": "coinbase"},
        {"symbol": "XRP/EUR", "name": "Ripple", "exchange": "coinbase"},
        {"symbol": "DOGE/EUR", "name": "Dogecoin", "exchange": "coinbase"},
        {"symbol": "ETC/EUR", "name": "Ethereum Classic", "exchange": "coinbase"},
        
        # DeFi Tokens
        {"symbol": "UNI/EUR", "name": "Uniswap", "exchange": "coinbase"},
        {"symbol": "AAVE/EUR", "name": "Aave", "exchange": "coinbase"},
        {"symbol": "CRV/EUR", "name": "Curve DAO Token", "exchange": "coinbase"},
        {"symbol": "SNX/EUR", "name": "Synthetix", "exchange": "coinbase"},
        {"symbol": "1INCH/EUR", "name": "1inch", "exchange": "coinbase"},
        {"symbol": "ENS/EUR", "name": "Ethereum Name Service", "exchange": "coinbase"},
        
        # Layer 1 Smart Contract Platforms
        {"symbol": "SOL/EUR", "name": "Solana", "exchange": "coinbase"},
        {"symbol": "ADA/EUR", "name": "Cardano", "exchange": "coinbase"},
        {"symbol": "DOT/EUR", "name": "Polkadot", "exchange": "coinbase"},
        {"symbol": "AVAX/EUR", "name": "Avalanche", "exchange": "coinbase"},
        {"symbol": "ATOM/EUR", "name": "Cosmos", "exchange": "coinbase"},
        {"symbol": "ALGO/EUR", "name": "Algorand", "exchange": "coinbase"},
        {"symbol": "MINA/EUR", "name": "Mina", "exchange": "coinbase"},
        {"symbol": "XTZ/EUR", "name": "Tezos", "exchange": "coinbase"},
        
        # Utility Tokens
        {"symbol": "LINK/EUR", "name": "Chainlink", "exchange": "coinbase"},
        {"symbol": "BAT/EUR", "name": "Basic Attention Token", "exchange": "coinbase"},
        {"symbol": "CHZ/EUR", "name": "Chiliz", "exchange": "coinbase"},
        {"symbol": "MANA/EUR", "name": "Decentraland", "exchange": "coinbase"},
        {"symbol": "FIL/EUR", "name": "Filecoin", "exchange": "coinbase"},
        {"symbol": "GRT/EUR", "name": "The Graph", "exchange": "coinbase"},
        {"symbol": "ICP/EUR", "name": "Internet Computer", "exchange": "coinbase"},
        {"symbol": "MASK/EUR", "name": "Mask Network", "exchange": "coinbase"},
        {"symbol": "XLM/EUR", "name": "Stellar", "exchange": "coinbase"},
        {"symbol": "EOS/EUR", "name": "EOS", "exchange": "coinbase"},
        
        # Gaming/NFT Tokens
        {"symbol": "APE/EUR", "name": "ApeCoin", "exchange": "coinbase"},
        {"symbol": "AXS/EUR", "name": "Axie Infinity", "exchange": "coinbase"},
    
        
        # Layer 2/Scaling Solutions
        {"symbol": "MATIC/EUR", "name": "Polygon", "exchange": "coinbase"},
        {"symbol": "ANKR/EUR", "name": "Ankr", "exchange": "coinbase"},
        
        # Meme Coins
        {"symbol": "SHIB/EUR", "name": "Shiba Inu", "exchange": "coinbase"},
        
        # Stablecoins
        {"symbol": "USDC/EUR", "name": "USD Coin", "exchange": "coinbase"},
        {"symbol": "USDT/EUR", "name": "Tether", "exchange": "coinbase"},
        
        # Additional Tokens
        {"symbol": "CGLD/EUR", "name": "Celo", "exchange": "coinbase"},
        {"symbol": "CRO/EUR", "name": "Cronos", "exchange": "coinbase"},
    ],
    "coinbase_majors_usd": [
        {"symbol": "BTC/USDT", "name": "Bitcoin", "exchange": "coinbase", "allocation": 0.40},
        {"symbol": "ETH/USDT", "name": "Ethereum", "exchange": "coinbase", "allocation": 0.30},
        {"symbol": "SOL/USDT", "name": "Solana", "exchange": "coinbase", "allocation": 0.20},
        {"symbol": "ADA/USDT", "name": "Cardano", "exchange": "coinbase", "allocation": 0.10},
    ],
}

# Trading strategies
TRADING_STRATEGIES = {
    "conservative": {
        "max_position_size": 0.05,  # 5% max per position
        "stop_loss": 0.10,          # 10% stop loss
        "take_profit": 0.20,        # 20% take profit
        "max_daily_trades": 3,
        "risk_per_trade": 0.02,     # 2% risk per trade
    },
    "moderate": {
        "max_position_size": 0.10,  # 10% max per position
        "stop_loss": 0.15,          # 15% stop loss
        "take_profit": 0.30,        # 30% take profit
        "max_daily_trades": 5,
        "risk_per_trade": 0.03,     # 3% risk per trade
    },
    "aggressive": {
        "max_position_size": 0.20,  # 20% max per position
        "stop_loss": 0.25,          # 25% stop loss
        "take_profit": 0.50,        # 50% take profit
        "max_daily_trades": 10,
        "risk_per_trade": 0.05,     # 5% risk per trade
    },
    "scalping": {
        "max_position_size": 0.05,  # 5% max per position
        "stop_loss": 0.05,          # 5% stop loss
        "take_profit": 0.10,        # 10% take profit
        "max_daily_trades": 20,
        "risk_per_trade": 0.01,     # 1% risk per trade
        "min_holding_time": 300,    # 5 minutes minimum
    }
}

# Market conditions and filters
MARKET_FILTERS = {
    "min_volume_24h": 1000000,      # Minimum 24h volume in EUR
    "min_market_cap": 10000000,     # Minimum market cap in EUR
    "max_spread": 0.02,             # Maximum bid-ask spread (2%)
    "min_price": 0.01,              # Minimum price per token
    "max_price": 100000,            # Maximum price per token
}

# Time-based trading rules
TIME_RULES = {
    "trading_hours": {
        "start": "00:00",           # 24-hour trading for crypto
        "end": "23:59",
        "timezone": "UTC"
    },
    "avoid_weekends": False,        # Crypto trades 24/7
    "avoid_holidays": False,        # Crypto trades on holidays
    "rebalance_frequency": "weekly", # How often to rebalance portfolio
}

def get_portfolio(portfolio_name: str = "coinbase_majors"):
    """Get a specific portfolio configuration"""
    return ASSET_PORTFOLIOS.get(portfolio_name, ASSET_PORTFOLIOS["coinbase_majors"])

def get_strategy(strategy_name: str = "moderate"):
    """Get a specific trading strategy configuration"""
    return TRADING_STRATEGIES.get(strategy_name, TRADING_STRATEGIES["moderate"])

def list_available_portfolios():
    """List all available portfolio names"""
    return list(ASSET_PORTFOLIOS.keys())

def list_available_strategies():
    """List all available strategy names"""
    return list(TRADING_STRATEGIES.keys())

def validate_portfolio(portfolio):
    """Validate portfolio structure"""
    if not portfolio:
        return False
    
    # Check if portfolio has allocations (old style) or not (new LLM-driven style)
    has_allocations = all("allocation" in asset for asset in portfolio)
    
    if has_allocations:
        # Old style validation: check allocations sum to 1.0
        total_allocation = sum(asset["allocation"] for asset in portfolio)
        return abs(total_allocation - 1.0) < 0.01  # Allow small rounding errors
    else:
        # New style validation: just check required fields exist
        required_fields = ["symbol", "name", "exchange"]
        return all(all(field in asset for field in required_fields) for asset in portfolio)

if __name__ == "__main__":
    # Test the configuration
    print("Available Portfolios:")
    for name in list_available_portfolios():
        portfolio = get_portfolio(name)
        valid = validate_portfolio(portfolio)
        print(f"  {name}: {len(portfolio)} assets, valid: {valid}")
    
    print("\nAvailable Strategies:")
    for name in list_available_strategies():
        strategy = get_strategy(name)
        print(f"  {name}: max_position_size={strategy['max_position_size']}, risk_per_trade={strategy['risk_per_trade']}") 