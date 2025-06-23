# AI Trading Bot with Local LLM

A sophisticated trading bot that uses a local Large Language Model (LLM) to make trading decisions. The bot connects to cryptocurrency exchanges, analyzes market data, and executes trades based on AI-generated decisions.

## Features

- 🤖 **Local LLM Integration**: Works with Ollama, LM Studio, or any OpenAI-compatible local model
- 📊 **Real-time Market Data**: Fetches live market data with technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
- 💰 **Risk Management**: Built-in stop-loss, take-profit, and position sizing
- 📈 **Paper Trading**: Test strategies without real money
- 🔄 **Automated Trading**: Runs continuously with configurable intervals
- 📝 **Comprehensive Logging**: Detailed logs and trade history
- 🛡️ **Safety Features**: Daily loss limits, portfolio risk management

## Prerequisites

1. **Python 3.8+**
2. **Local LLM Server** (Ollama, LM Studio, etc.)
3. **Exchange API Keys** (for real trading)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd trading-bot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your local LLM:**
   
   **Option A: Ollama (Recommended)**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama2
   ollama serve
   ```
   
   **Option B: LM Studio**
   - Download from https://lmstudio.ai
   - Load a model and start the local server
   
   **Option C: Other local models**
   - Any OpenAI-compatible API server

4. **Configure the bot:**
   ```bash
   cp env_example.txt .env
   # Edit .env with your settings
   ```

## Configuration

Edit the `.env` file with your settings:

```env
# LLM Configuration
LLM_BASE_URL=http://localhost:11434  # Ollama default
LLM_MODEL=llama2                     # Your model name
LLM_API_KEY=                         # Optional API key

# Trading Configuration
EXCHANGE=binance                     # Exchange name
SYMBOL=BTC/USDT                      # Trading pair
TIMEFRAME=1h                         # Data timeframe
POSITION_SIZE=0.01                   # 1% of portfolio per trade
MAX_POSITIONS=3                      # Maximum concurrent positions
STOP_LOSS_PCT=0.02                  # 2% stop loss
TAKE_PROFIT_PCT=0.04                # 4% take profit

# Exchange API Keys (for real trading)
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_SECRET=your_secret_here
EXCHANGE_PASSPHRASE=your_passphrase_here

# Risk Management
MAX_DAILY_LOSS=0.05                 # 5% daily loss limit
MAX_PORTFOLIO_RISK=0.02             # 2% portfolio risk

# Bot Settings
TRADING_ENABLED=false               # Set to true for real trading
PAPER_TRADING=true                  # Use paper trading by default
LOG_LEVEL=INFO
```

## Usage

### Command Line Interface

The bot includes a CLI for easy management:

```bash
# Test LLM connection
python cli.py test-llm

# Test market data fetching
python cli.py test-data

# Run a single trading cycle
python cli.py cycle

# Check bot status
python cli.py status

# Start the bot (runs continuously)
python cli.py start --interval 60  # 60-minute intervals
```

### Python API

```python
import asyncio
from trading_bot import TradingBot

async def main():
    bot = TradingBot()
    
    # Initialize and run
    if await bot.initialize():
        await bot.start(interval_minutes=60)
    
    # Or run a single cycle
    await bot.run_trading_cycle()
    
    # Get status
    status = bot.get_status()
    print(status)

asyncio.run(main())
```

## How It Works

1. **Data Collection**: Fetches real-time market data including price, volume, and technical indicators
2. **AI Analysis**: Sends market data to your local LLM for analysis
3. **Decision Making**: LLM returns trading decisions (BUY/SELL/HOLD/CLOSE)
4. **Risk Management**: Checks stop-losses, position limits, and risk parameters
5. **Trade Execution**: Executes trades on the exchange (or simulates in paper trading)
6. **Monitoring**: Logs all activities and maintains trade history

## Supported Exchanges

The bot uses the CCXT library and supports 100+ exchanges including:
- Binance
- Coinbase Pro
- Kraken
- Bitfinex
- And many more...

## LLM Integration

The bot supports multiple LLM formats:

### Ollama Format
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "prompt": "Hello"}'
```

### OpenAI-Compatible Format
```bash
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local-model", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Anthropic Format
```bash
curl -X POST http://localhost:1234/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Safety Features

- **Paper Trading**: Test strategies without real money
- **Position Limits**: Maximum number of concurrent positions
- **Stop Losses**: Automatic position closure at loss thresholds
- **Take Profits**: Automatic position closure at profit targets
- **Daily Loss Limits**: Stop trading if daily loss exceeds threshold
- **Portfolio Risk Management**: Limit total exposure

## Logging

Logs are stored in `logs/trading_bot.log` with rotation:
- Daily log rotation
- 7-day retention
- Configurable log levels

## Trade History

The bot maintains detailed trade history including:
- Market data at time of trade
- LLM decision and reasoning
- Trade execution results
- Portfolio performance

Save trade history:
```python
bot.save_trade_history("my_trades.json")
```

## Development

### Project Structure
```
trading-bot/
├── config.py          # Configuration management
├── llm_client.py      # LLM integration
├── data_fetcher.py    # Market data collection
├── trading_engine.py  # Trade execution
├── trading_bot.py     # Main bot orchestration
├── cli.py            # Command line interface
├── requirements.txt   # Dependencies
└── logs/             # Log files
```

### Adding New Features

1. **New Technical Indicators**: Add to `data_fetcher.py`
2. **New Exchanges**: CCXT handles most exchanges automatically
3. **New LLM Providers**: Extend `llm_client.py`
4. **New Trading Strategies**: Modify the LLM prompt in `llm_client.py`

## Troubleshooting

### Common Issues

1. **LLM Connection Failed**
   - Check if your LLM server is running
   - Verify the URL in `LLM_BASE_URL`
   - Test with `python cli.py test-llm`

2. **Market Data Issues**
   - Check internet connection
   - Verify exchange API keys
   - Test with `python cli.py test-data`

3. **Trading Errors**
   - Ensure `TRADING_ENABLED=true` for real trading
   - Check exchange API permissions
   - Verify account has sufficient funds

### Debug Mode

Set log level to DEBUG in `.env`:
```env
LOG_LEVEL=DEBUG
```

## Disclaimer

⚠️ **This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Never trade with money you cannot afford to lose. The authors are not responsible for any financial losses.**

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/trading_bot.log`
3. Open an issue on GitHub 