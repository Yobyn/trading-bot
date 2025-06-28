import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class TradingConfig(BaseSettings):
    # LLM Configuration
    llm_base_url: str = "http://localhost:11434"  # Default Ollama URL
    llm_model: str = "llama2"
    llm_api_key: Optional[str] = None
    
    # Trading Configuration
    exchange: str = "coinbase"
    symbol: str = "BTC/EUR"
    timeframe: str = "1h"
    position_size: float = 0.01  # 1% of portfolio
    max_positions: int = 3
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    
    # API Keys (set in .env file)
    exchange_api_key: Optional[str] = None
    exchange_secret: Optional[str] = None
    exchange_passphrase: Optional[str] = None

    # Coinbase API Keys (set in .env file)
    coinbase_api_key: Optional[str] = None
    coinbase_secret: Optional[str] = None
    coinbase_passphrase: Optional[str] = None
    
    # Risk Management
    max_daily_loss: float = 0.05  # 5%
    max_portfolio_risk: float = 0.02  # 2%
    
    # Bot Settings
    trading_enabled: bool = False  # Set to True to enable live trading
    paper_trading: bool = True  # Use paper trading by default
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields from .env

config = TradingConfig() 