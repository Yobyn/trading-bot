# AI Trading Bot

An intelligent trading bot that uses local LLM (Ollama) for market analysis and decision making. Supports multiple assets, portfolios, and trading strategies.

## Features

- 🤖 **AI-Powered Decisions**: Uses local LLM (Ollama) for market analysis
- 📈 **Multi-Asset Trading**: Trade multiple cryptocurrencies simultaneously
- 🎯 **Portfolio Management**: Pre-configured portfolios for different asset classes
- ⚡ **Trading Strategies**: Conservative, moderate, aggressive, and scalping strategies
- 📊 **Real-time Data**: Live market data from Binance
- 🧪 **Paper Trading**: Safe testing with virtual money
- 📝 **Comprehensive Logging**: Detailed logs and analysis reports
- 🔧 **Easy Configuration**: Simple setup and customization

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Install Ollama

```bash
# Install Ollama
brew install ollama

# Pull a model (llama2 recommended)
ollama pull llama2

# Start Ollama server
ollama serve
```

### 3. Configure Environment

```bash
# Copy environment template
cp env_template.txt .env

# Edit .env with your settings
nano .env
```

### 4. Start Trading

#### Simple Interactive Mode
```bash
python start_trading.py
```

#### Command Line Mode
```bash
# List available options
python enhanced_cli.py list

# Test with specific portfolio and strategy
python enhanced_cli.py test --portfolio crypto_majors --strategy moderate

# Start trading
python enhanced_cli.py start --portfolio defi_tokens --strategy conservative --interval 15
```

## Available Portfolios

### 📈 Crypto Majors
- Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB)
- Solana (SOL), Cardano (ADA), Polkadot (DOT), Chainlink (LINK)

### 🏦 DeFi Tokens
- Uniswap (UNI), Aave (AAVE), Compound (COMP)
- SushiSwap (SUSHI), Curve (CRV), Yearn Finance (YFI)

### ⛓️ Layer 1 Blockchains
- Ethereum (ETH), Solana (SOL), Avalanche (AVAX)
- Cosmos (ATOM), NEAR Protocol (NEAR), Fantom (FTM), Algorand (ALGO)

### 🎮 Gaming Tokens
- Axie Infinity (AXS), The Sandbox (SAND), Decentraland (MANA)
- Enjin Coin (ENJ), Gala (GALA)

### 🤖 AI Tokens
- Fetch.ai (FET), Ocean Protocol (OCEAN), SingularityNET (AGIX)
- Render Token (RNDR), Basic Attention Token (BAT)

### 🐕 Meme Coins
- Dogecoin (DOGE), Shiba Inu (SHIB), Pepe (PEPE), Floki (FLOKI)

### 🎯 Custom Portfolio
- Customizable asset allocation

## Trading Strategies

### 🛡️ Conservative
- Max position size: 5%
- Stop loss: 10%
- Take profit: 20%
- Max daily trades: 3
- Risk per trade: 2%

### ⚖️ Moderate (Default)
- Max position size: 10%
- Stop loss: 15%
- Take profit: 30%
- Max daily trades: 5
- Risk per trade: 3%

### 🚀 Aggressive
- Max position size: 20%
- Stop loss: 25%
- Take profit: 50%
- Max daily trades: 10
- Risk per trade: 5%

### ⚡ Scalping
- Max position size: 5%
- Stop loss: 5%
- Take profit: 10%
- Max daily trades: 20
- Risk per trade: 1%
- Min holding time: 5 minutes

## Configuration

### Environment Variables (.env)

```bash
# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama2
LLM_BASE_URL=http://localhost:11434

# Trading Configuration
TRADING_ENABLED=false
PAPER_TRADING=true
INITIAL_BALANCE=10000

# Exchange Configuration
EXCHANGE=binance
API_KEY=your_api_key
API_SECRET=your_api_secret

# Logging
LOG_LEVEL=INFO
```

### Custom Portfolios

Edit `asset_config.py` to create custom portfolios:

```python
"my_custom_portfolio": [
    {"symbol": "BTC/USDT", "name": "Bitcoin", "exchange": "binance", "allocation": 0.50},
    {"symbol": "ETH/USDT", "name": "Ethereum", "exchange": "binance", "allocation": 0.30},
    {"symbol": "SOL/USDT", "name": "Solana", "exchange": "binance", "allocation": 0.20},
]
```

## Usage Examples

### Basic Single Asset Trading
```bash
# Start with Bitcoin only
python cli.py start --interval 15
```

### Multi-Asset Trading
```bash
# Start with crypto majors portfolio
python enhanced_cli.py start --portfolio crypto_majors --strategy moderate

# Test DeFi tokens with conservative strategy
python enhanced_cli.py test --portfolio defi_tokens --strategy conservative

# Start aggressive AI tokens trading
python enhanced_cli.py start --portfolio ai_tokens --strategy aggressive --interval 10
```

### Interactive Mode
```bash
python start_trading.py
# Follow the prompts to select portfolio, strategy, and interval
```

## File Structure

```
trading-bot/
├── config.py              # Configuration management
├── llm_client.py          # LLM integration (Ollama/OpenAI)
├── data_fetcher.py        # Market data fetching
├── trading_engine.py      # Order execution and risk management
├── trading_bot.py         # Single asset trading bot
├── multi_asset_bot.py     # Basic multi-asset bot
├── enhanced_multi_bot.py  # Enhanced multi-asset bot with strategies
├── asset_config.py        # Portfolio and strategy definitions
├── cli.py                 # Single asset CLI
├── multi_asset_cli.py     # Multi-asset CLI
├── enhanced_cli.py        # Enhanced CLI with portfolio/strategy selection
├── start_trading.py       # Interactive trading starter
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from env_template.txt)
└── logs/                  # Log files
```

## Logging and Analysis

The bot generates detailed logs and analysis reports:

- **Log files**: `logs/enhanced_multi_bot_{portfolio}_{strategy}.log`
- **Analysis reports**: `enhanced_analysis_{portfolio}_{strategy}_{timestamp}.json`
- **Portfolio tracking**: Real-time portfolio value and P&L

## Safety Features

- ✅ **Paper Trading**: Test strategies without real money
- ✅ **Position Sizing**: Automatic risk management
- ✅ **Stop Losses**: Built-in loss protection
- ✅ **Daily Limits**: Maximum trades per day
- ✅ **Portfolio Diversification**: Multiple asset allocation

## Supported Exchanges

- **Binance**: Full support (recommended)
- **Alpaca**: Stock trading (requires API keys)
- **CCXT**: 100+ exchanges via CCXT library

## Troubleshooting

### LLM Connection Issues
```bash
# Check if Ollama is running
ollama list

# Restart Ollama
ollama serve
```

### Data Fetching Issues
```bash
# Test data connection
python -c "from data_fetcher import DataFetcher; import asyncio; asyncio.run(DataFetcher().get_market_data())"
```

### Portfolio Issues
```bash
# Validate portfolio configuration
python asset_config.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Disclaimer

This is educational software. Trading involves risk and you can lose money. Always:
- Start with paper trading
- Understand the strategies before using real money
- Never invest more than you can afford to lose
- Consider consulting a financial advisor

## License

MIT License - see LICENSE file for details. 