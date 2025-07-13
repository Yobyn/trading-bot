# Performance Monitoring System

## Overview

The Performance Monitoring System provides comprehensive tracking and analysis of trading bot performance, including:

- **Trade Tracking**: Every buy/sell decision with market data
- **Portfolio Snapshots**: Portfolio value, positions, and P&L over time
- **Performance Metrics**: Returns, Sharpe ratio, drawdown, win rate, etc.
- **Advanced Analytics**: Charts, comparisons, and detailed reports

## Features

### 📊 **Comprehensive Metrics**
- Total and annualized returns
- Sharpe ratio and volatility
- Maximum drawdown
- Win rate and profit factor
- Average win/loss ratios
- Best and worst trades

### 📈 **Trade Analysis**
- Individual trade records with market data
- Technical indicators (RSI, MACD, volume)
- Profit/loss tracking
- Confidence levels and decision reasoning

### 💼 **Portfolio Tracking**
- Real-time portfolio value
- Cash vs positions allocation
- Position count and diversity
- Daily P&L tracking

### 📋 **Reporting System**
- Detailed performance reports
- Time-based comparisons
- Export to CSV/JSON
- Visual charts and graphs

## Installation

The performance monitoring system is automatically integrated into the trading bot. No additional installation required.

### Optional Dependencies

For charting functionality, install:
```bash
pip install matplotlib seaborn
```

## Usage

### Command Line Interface

The performance CLI provides easy access to all monitoring features:

```bash
# Display performance summary
python performance_cli.py summary --days 30

# Generate detailed report
python performance_cli.py report --days 30 --save

# View trading statistics
python performance_cli.py trades --days 30 --symbol BTC/EUR

# Show portfolio history
python performance_cli.py portfolio --days 30

# Generate performance charts
python performance_cli.py chart --days 30 --show

# Compare performance across time periods
python performance_cli.py compare

# Export data to CSV
python performance_cli.py export --days 30
```

### Available Commands

#### **Summary**
```bash
python performance_cli.py summary [--days N]
```
Display key performance metrics for the specified period.

#### **Report**
```bash
python performance_cli.py report [--days N] [--save]
```
Generate comprehensive performance report with recent trades and portfolio evolution.

#### **Trades**
```bash
python performance_cli.py trades [--days N] [--symbol SYMBOL]
```
Show trading statistics, optionally filtered by symbol.

#### **Portfolio**
```bash
python performance_cli.py portfolio [--days N]
```
Display portfolio value history and allocation changes.

#### **Chart**
```bash
python performance_cli.py chart [--days N] [--show]
```
Generate visual performance charts (requires matplotlib/seaborn).

#### **Compare**
```bash
python performance_cli.py compare
```
Compare performance across 7, 30, and 90-day periods.

#### **Export**
```bash
python performance_cli.py export [--days N]
```
Export performance data to CSV files.

## Data Storage

### Database Structure

Performance data is stored in SQLite database (`performance_data/performance.db`):

- **trades**: Individual trade records
- **portfolio_snapshots**: Portfolio state over time
- **performance_metrics**: Calculated metrics history

### File Organization

```
performance_data/
├── performance.db              # SQLite database
├── performance_report_*.json   # Generated reports
└── performance_chart_*.png     # Generated charts
```

## Integration with Trading Bot

The performance monitoring system is automatically integrated with the enhanced multi-bot:

### Automatic Recording

- **Trade Execution**: Every buy/sell order is recorded
- **Portfolio Updates**: Portfolio snapshots after each trading cycle
- **Market Data**: Technical indicators and market context

### Real-time Monitoring

- Performance metrics calculated in real-time
- Portfolio tracking with each cycle
- Audit trail integration for complete visibility

## Performance Metrics Explained

### **Returns**
- **Total Return**: Overall portfolio performance
- **Annualized Return**: Yearly performance projection
- **Daily Returns**: Day-to-day performance variations

### **Risk Metrics**
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns

### **Trading Metrics**
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of total profits to total losses
- **Average Win/Loss**: Average profit per winning/losing trade

### **Position Metrics**
- **Total Trades**: Number of buy/sell operations
- **Best/Worst Trade**: Highest profit and loss trades
- **Average Trade Duration**: Time positions are held

## Example Output

### Performance Summary
```
============================================================
PERFORMANCE SUMMARY - Last 30 Days
============================================================
Total Return:        +12.45%
Annualized Return:   +152.34%
Sharpe Ratio:        1.85
Max Drawdown:        -3.21%
Volatility:          15.67%
------------------------------------------------------------
Total Trades:        47
Win Rate:            68.1%
Winning Trades:      32
Losing Trades:       15
Profit Factor:       2.14
------------------------------------------------------------
Average Win:         €24.50
Average Loss:        €11.25
Best Trade:          €85.30
Worst Trade:         €-28.15
============================================================
```

### Trading Statistics
```
============================================================
TRADING STATISTICS
============================================================

BTC/EUR:
--------------------
  BUY :  12 trades | Avg P&L: €+15.20 | Total: €+182.40 | Confidence: 78.5%
  SELL:   8 trades | Avg P&L: €+22.10 | Total: €+176.80 | Confidence: 82.1%
  HOLD:  15 trades | Avg P&L: €+0.00  | Total: €+0.00   | Confidence: 65.2%

ETH/EUR:
--------------------
  BUY :   9 trades | Avg P&L: €+8.45  | Total: €+76.05  | Confidence: 75.3%
  SELL:   6 trades | Avg P&L: €+12.30 | Total: €+73.80  | Confidence: 79.8%
  HOLD:  12 trades | Avg P&L: €+0.00  | Total: €+0.00   | Confidence: 62.1%
```

## Troubleshooting

### Common Issues

**No Data Available**
- Ensure the trading bot has been running and making trades
- Check that the database file exists in `performance_data/`

**Chart Generation Fails**
- Install required dependencies: `pip install matplotlib seaborn`
- Ensure sufficient data for the requested time period

**Database Errors**
- Check file permissions in `performance_data/` directory
- Verify SQLite database is not corrupted

### Debug Mode

Enable debug logging to troubleshoot issues:
```bash
export LOG_LEVEL=DEBUG
python performance_cli.py summary
```

## Advanced Usage

### Custom Analysis

Use the PerformanceMonitor class directly for custom analysis:

```python
from performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
metrics = monitor.calculate_performance_metrics(days=30)
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

### Data Export

Export raw data for external analysis:

```python
history = monitor.get_portfolio_history(days=30)
trades = monitor.get_trading_statistics(days=30)
```

## Future Enhancements

Planned improvements include:

- **Machine Learning**: Predictive performance modeling
- **Benchmarking**: Compare against market indices
- **Risk Attribution**: Identify sources of risk and return
- **Real-time Alerts**: Notifications for performance thresholds
- **Web Dashboard**: Interactive performance visualization

## Support

For issues or questions about the performance monitoring system:

1. Check the logs in `logs/performance_cli.log`
2. Verify data availability and time periods
3. Ensure all dependencies are installed
4. Review the troubleshooting section above

The performance monitoring system provides comprehensive insights into trading bot performance, helping optimize strategies and track progress over time. 