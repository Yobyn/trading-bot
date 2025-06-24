#!/usr/bin/env python3
"""
Asset Configuration for Multi-Asset Trading Bot
"""

# Predefined asset portfolios
ASSET_PORTFOLIOS = {
    "crypto_majors": [
        {"symbol": "BTC/USDT", "name": "Bitcoin", "exchange": "binance", "allocation": 0.25},
        {"symbol": "ETH/USDT", "name": "Ethereum", "exchange": "binance", "allocation": 0.20},
        {"symbol": "BNB/USDT", "name": "Binance Coin", "exchange": "binance", "allocation": 0.15},
        {"symbol": "SOL/USDT", "name": "Solana", "exchange": "binance", "allocation": 0.15},
        {"symbol": "ADA/USDT", "name": "Cardano", "exchange": "binance", "allocation": 0.10},
        {"symbol": "DOT/USDT", "name": "Polkadot", "exchange": "binance", "allocation": 0.10},
        {"symbol": "LINK/USDT", "name": "Chainlink", "exchange": "binance", "allocation": 0.05},
    ],
    
    "defi_tokens": [
        {"symbol": "UNI/USDT", "name": "Uniswap", "exchange": "binance", "allocation": 0.20},
        {"symbol": "AAVE/USDT", "name": "Aave", "exchange": "binance", "allocation": 0.20},
        {"symbol": "COMP/USDT", "name": "Compound", "exchange": "binance", "allocation": 0.15},
        {"symbol": "SUSHI/USDT", "name": "SushiSwap", "exchange": "binance", "allocation": 0.15},
        {"symbol": "CRV/USDT", "name": "Curve", "exchange": "binance", "allocation": 0.15},
        {"symbol": "YFI/USDT", "name": "Yearn Finance", "exchange": "binance", "allocation": 0.15},
    ],
    
    "layer1_blockchains": [
        {"symbol": "ETH/USDT", "name": "Ethereum", "exchange": "binance", "allocation": 0.25},
        {"symbol": "SOL/USDT", "name": "Solana", "exchange": "binance", "allocation": 0.20},
        {"symbol": "AVAX/USDT", "name": "Avalanche", "exchange": "binance", "allocation": 0.15},
        {"symbol": "ATOM/USDT", "name": "Cosmos", "exchange": "binance", "allocation": 0.15},
        {"symbol": "NEAR/USDT", "name": "NEAR Protocol", "exchange": "binance", "allocation": 0.10},
        {"symbol": "FTM/USDT", "name": "Fantom", "exchange": "binance", "allocation": 0.10},
        {"symbol": "ALGO/USDT", "name": "Algorand", "exchange": "binance", "allocation": 0.05},
    ],
    
    "meme_coins": [
        {"symbol": "DOGE/USDT", "name": "Dogecoin", "exchange": "binance", "allocation": 0.30},
        {"symbol": "SHIB/USDT", "name": "Shiba Inu", "exchange": "binance", "allocation": 0.30},
        {"symbol": "PEPE/USDT", "name": "Pepe", "exchange": "binance", "allocation": 0.20},
        {"symbol": "FLOKI/USDT", "name": "Floki", "exchange": "binance", "allocation": 0.20},
    ],
    
    "gaming_tokens": [
        {"symbol": "AXS/USDT", "name": "Axie Infinity", "exchange": "binance", "allocation": 0.25},
        {"symbol": "SAND/USDT", "name": "The Sandbox", "exchange": "binance", "allocation": 0.25},
        {"symbol": "MANA/USDT", "name": "Decentraland", "exchange": "binance", "allocation": 0.20},
        {"symbol": "ENJ/USDT", "name": "Enjin Coin", "exchange": "binance", "allocation": 0.15},
        {"symbol": "GALA/USDT", "name": "Gala", "exchange": "binance", "allocation": 0.15},
    ],
    
    "ai_tokens": [
        {"symbol": "FET/USDT", "name": "Fetch.ai", "exchange": "binance", "allocation": 0.25},
        {"symbol": "OCEAN/USDT", "name": "Ocean Protocol", "exchange": "binance", "allocation": 0.25},
        {"symbol": "AGIX/USDT", "name": "SingularityNET", "exchange": "binance", "allocation": 0.20},
        {"symbol": "RNDR/USDT", "name": "Render Token", "exchange": "binance", "allocation": 0.15},
        {"symbol": "BAT/USDT", "name": "Basic Attention Token", "exchange": "binance", "allocation": 0.15},
    ],
    
    "custom_portfolio": [
        # Add your custom assets here
        {"symbol": "BTC/USDT", "name": "Bitcoin", "exchange": "binance", "allocation": 0.40},
        {"symbol": "ETH/USDT", "name": "Ethereum", "exchange": "binance", "allocation": 0.30},
        {"symbol": "SOL/USDT", "name": "Solana", "exchange": "binance", "allocation": 0.20},
        {"symbol": "ADA/USDT", "name": "Cardano", "exchange": "binance", "allocation": 0.10},
    ]
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
    "min_volume_24h": 1000000,      # Minimum 24h volume in USD
    "min_market_cap": 10000000,     # Minimum market cap in USD
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

def get_portfolio(portfolio_name: str = "crypto_majors"):
    """Get a specific portfolio configuration"""
    return ASSET_PORTFOLIOS.get(portfolio_name, ASSET_PORTFOLIOS["crypto_majors"])

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
    """Validate that portfolio allocations sum to 1.0"""
    total_allocation = sum(asset["allocation"] for asset in portfolio)
    return abs(total_allocation - 1.0) < 0.01  # Allow small rounding errors

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