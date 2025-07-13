# Web Dashboard for Trading Bot

## Overview

The web dashboard provides a real-time monitoring and control interface for your trading bot with comprehensive analytics, performance metrics, and risk management data.

## Features

### 🎛️ **Bot Control Panel**
- **Start/Stop Bot**: Control bot execution with custom parameters
- **Portfolio Selection**: Choose from different crypto portfolios
- **Strategy Selection**: Pick trading strategies (conservative, moderate, aggressive, scalping)
- **Interval Configuration**: Set trading cycle intervals (1-1440 minutes)
- **Real-time Status**: Monitor bot status and errors

### 📊 **Real-time Metrics**
- **Portfolio Value**: Current total portfolio value
- **Total P&L**: Overall profit/loss across all positions
- **Active Positions**: Number of currently held positions
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Largest portfolio decline
- **Volatility**: Portfolio price volatility

### 📈 **Interactive Charts**
- **Portfolio Value Chart**: Historical portfolio value over time
- **Risk Metrics Chart**: Visual representation of VaR, CVaR, concentration risk, and volatility
- **Real-time Updates**: Charts update automatically every 30 seconds

### 📋 **Data Tables**
- **Current Positions**: All active positions with P&L
- **Recent Trades**: Latest buy/sell transactions
- **Trading Decisions**: LLM decisions with reasoning and market data

### 🔄 **Real-time Updates**
- **WebSocket Connection**: Live data streaming
- **30-Second Refresh**: Automatic data updates
- **Connection Status**: Visual connection indicator

## Installation

### 1. Install Dependencies

```bash
pip install flask flask-socketio plotly
```

### 2. Verify Installation

```bash
python -c "import flask, flask_socketio, plotly; print('All dependencies installed successfully')"
```

## Usage

### Quick Start

```bash
# Start dashboard on localhost:5000
python dashboard_cli.py

# Start with custom port
python dashboard_cli.py --port 8080

# Start with debug mode
python dashboard_cli.py --debug

# Start accessible from all network interfaces
python dashboard_cli.py --public --port 5000
```

### Direct Python Usage

```python
from web_dashboard import start_dashboard

# Start dashboard
start_dashboard(host='127.0.0.1', port=5000, debug=False)
```

### Access Dashboard

Open your browser and navigate to:
- Local: `http://localhost:5000`
- Network: `http://YOUR_IP:5000` (if using --public)

## Dashboard Sections

### 1. Control Panel

**Bot Control Options:**
- **Portfolio**: Select crypto portfolio to trade
  - `coinbase_all_eur`: All EUR-paired cryptocurrencies
  - `coinbase_majors`: Major cryptocurrencies
  - `coinbase_majors_usd`: Major USD-paired cryptocurrencies
- **Strategy**: Choose trading strategy
  - `conservative`: Low-risk, stable returns
  - `moderate`: Balanced risk/reward
  - `aggressive`: Higher risk, higher potential returns
  - `scalping`: Short-term trading
- **Interval**: Set trading cycle frequency (1-1440 minutes)

**Status Display:**
- Current bot status (Running/Stopped)
- Error messages (if any)
- Last update timestamp

### 2. Key Metrics Cards

**Portfolio Metrics:**
- **Portfolio Value**: Total account value in EUR
- **Total P&L**: Overall profit/loss
- **Active Positions**: Number of held positions
- **Win Rate**: Percentage of profitable trades

**Performance Metrics:**
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Largest portfolio decline
- **Volatility**: Portfolio price volatility

### 3. Interactive Charts

**Portfolio Value Chart:**
- Historical portfolio value over time
- Plotly interactive chart with zoom/pan
- Updates automatically with new data

**Risk Metrics Chart:**
- Visual representation of risk metrics
- VaR (Value at Risk)
- CVaR (Conditional Value at Risk)
- Concentration Risk
- Volatility

### 4. Data Tables

**Current Positions:**
- Symbol, Amount, Value, P&L for each position
- Color-coded profit/loss indicators
- Real-time updates

**Recent Trades:**
- Last 10 trades with timestamps
- Action badges (BUY/SELL/HOLD)
- Trade values and P&L

**Trading Decisions:**
- LLM decisions with reasoning
- Market data at decision time (price, RSI)
- Success/failure indicators

## API Endpoints

The dashboard provides REST API endpoints for programmatic access:

### Portfolio Data
```
GET /api/portfolio
```
Returns current portfolio summary.

### Performance Metrics
```
GET /api/performance?days=30
```
Returns performance metrics for specified days.

### Risk Metrics
```
GET /api/risk
```
Returns current risk assessment.

### Recent Trades
```
GET /api/trades?limit=20
```
Returns recent trades.

### Chart Data
```
GET /api/chart?days=30
```
Returns portfolio chart data.

### Bot Control
```
POST /api/bot/start
Content-Type: application/json

{
  "portfolio_name": "coinbase_all_eur",
  "strategy_name": "moderate",
  "interval_minutes": 15
}
```

```
POST /api/bot/stop
```

```
GET /api/bot/status
```

## Real-time Features

### WebSocket Events

The dashboard uses WebSocket for real-time updates:

- `portfolio_update`: Portfolio metrics update
- `performance_update`: Performance metrics update
- `risk_update`: Risk metrics update
- `connect`: Connection established
- `disconnect`: Connection lost

### Auto-refresh

- **Real-time**: WebSocket updates every 30 seconds
- **Fallback**: HTTP polling every 30 seconds
- **Connection Status**: Visual indicator in navbar

## Security Considerations

### Local Access (Default)
- Dashboard binds to `127.0.0.1` by default
- Only accessible from local machine
- Safe for development and personal use

### Network Access (--public flag)
- Binds to `0.0.0.0` - accessible from any network interface
- **⚠️ WARNING**: No authentication implemented
- **⚠️ WARNING**: Bot control endpoints are publicly accessible
- Only use on trusted networks
- Consider implementing authentication for production use

### Recommendations
- Use firewall rules to restrict access
- Consider VPN for remote access
- Implement authentication for production deployment
- Use HTTPS in production environments

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
ImportError: No module named 'flask'
```
**Solution**: Install dependencies
```bash
pip install flask flask-socketio plotly
```

**2. Port Already in Use**
```bash
OSError: [Errno 48] Address already in use
```
**Solution**: Use different port
```bash
python dashboard_cli.py --port 8080
```

**3. Connection Refused**
```bash
Connection refused when accessing dashboard
```
**Solution**: Check if dashboard is running and port is correct

**4. No Data Displayed**
```bash
Dashboard shows "No data" for all metrics
```
**Solution**: 
- Start the trading bot first
- Wait for initial data collection
- Check bot logs for errors

### Debug Mode

Enable debug mode for detailed error information:
```bash
python dashboard_cli.py --debug
```

### Logs

Dashboard logs are integrated with the main bot logging system:
- Check `logs/` directory for log files
- Look for dashboard-related errors
- Enable debug mode for verbose logging

## Integration with Bot Components

### Performance Monitor
- Pulls data from `performance_monitor.py`
- Displays metrics from SQLite database
- Real-time trade recording

### Risk Manager
- Integrates with `risk_manager.py`
- Shows portfolio risk metrics
- Real-time risk assessment

### Audit Trail
- Displays data from `audit_trail.py`
- Shows LLM decisions and reasoning
- Complete trading decision history

### Enhanced Multi-Bot
- Controls `enhanced_multi_bot.py`
- Start/stop bot operations
- Real-time status monitoring

## Customization

### Styling
- Bootstrap 5 for responsive design
- Custom CSS for trading bot theme
- Gradient backgrounds and modern UI

### Charts
- Plotly.js for interactive charts
- Customizable colors and layouts
- Responsive design for mobile devices

### Data Refresh
- Configurable refresh intervals
- WebSocket for real-time updates
- Graceful fallback to HTTP polling

## Future Enhancements

### Planned Features
- **Authentication**: User login system
- **Multi-user Support**: Multiple bot instances
- **Advanced Charts**: More technical indicators
- **Alerts**: Email/SMS notifications
- **Export**: Data export functionality
- **Mobile App**: React Native companion app

### Technical Improvements
- **Caching**: Redis for performance
- **Database**: PostgreSQL for production
- **Monitoring**: Prometheus metrics
- **Deployment**: Docker containerization

## Examples

### Basic Usage
```bash
# Start dashboard
python dashboard_cli.py

# Open browser to http://localhost:5000
# Click "Start Bot" to begin trading
# Monitor real-time metrics and charts
```

### Advanced Usage
```bash
# Start on custom port with debug
python dashboard_cli.py --port 8080 --debug

# Start accessible from network
python dashboard_cli.py --public --port 5000
```

### API Usage
```python
import requests

# Get portfolio data
response = requests.get('http://localhost:5000/api/portfolio')
portfolio = response.json()
print(f"Portfolio Value: €{portfolio['total_value']}")

# Start bot programmatically
requests.post('http://localhost:5000/api/bot/start', json={
    'portfolio_name': 'coinbase_all_eur',
    'strategy_name': 'moderate',
    'interval_minutes': 15
})
```

The web dashboard provides a comprehensive, real-time monitoring and control interface for your trading bot, making it easy to track performance, manage risk, and control bot operations from any web browser. 