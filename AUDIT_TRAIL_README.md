# Audit Trail System for Trading Bot

## Overview

The audit trail system captures comprehensive information about every trading decision made by the bot, including:

- **LLM Interactions**: Complete prompts, responses, and parsed decisions
- **Trading Decisions**: Market data, position information, and execution results
- **Portfolio Snapshots**: Portfolio value, positions, and P&L over time

## Directory Structure

```
audit_trails/
├── llm_interactions/          # Complete LLM prompts and responses
├── trading_decisions/         # Trading decisions with market data
└── portfolio_snapshots/       # Portfolio state over time
```

## What Gets Logged

### 1. LLM Interactions (`llm_interactions/`)

Each file contains:
- **System Prompt**: The instructions given to the LLM
- **User Prompt**: The market data and context provided
- **LLM Response**: The raw response from the LLM
- **Parsed Decision**: The extracted action (BUY/SELL/HOLD)
- **Market Data**: Complete market data at decision time
- **Trading Phase**: INVESTMENT or MANAGEMENT

Example filename: `llm_interaction_ETH_EUR_20250712_161612_123.json`

### 2. Trading Decisions (`trading_decisions/`)

Each file contains:
- **Action**: BUY, SELL, HOLD, BUY_FAILED, SELL_FAILED
- **Decision Reason**: Why the bot made this decision
- **Market Data**: RSI, price, averages, technical indicators
- **Position Data**: Current position size, P&L, buy price
- **Execution Result**: Success/failure, amounts, prices

Example filename: `trading_decision_ETH_EUR_BUY_20250712_161612_123.json`

### 3. Portfolio Snapshots (`portfolio_snapshots/`)

Each file contains:
- **Portfolio Value**: Total account value
- **Positions**: All current positions with P&L
- **Available Cash**: Cash available for trading
- **Total P&L**: Overall portfolio performance

Example filename: `portfolio_snapshot_20250712_161612_123.json`

## Analysis Tools

### 1. Analyze Trading Decisions

```bash
# Analyze all decisions from last 7 days
python analyze_audit_trail.py --type decisions --days 7

# Analyze decisions for specific symbol
python analyze_audit_trail.py --type decisions --symbol ETH/EUR --days 7
```

Output shows:
- Decision breakdown (BUY/SELL/HOLD percentages)
- Most traded symbols
- Recent decisions with RSI and price data
- Decision reasons

### 2. Analyze LLM Interactions

```bash
# Analyze LLM interactions from last 7 days
python analyze_audit_trail.py --type interactions --days 7

# Analyze interactions for specific symbol
python analyze_audit_trail.py --type interactions --symbol ETH/EUR --days 7
```

Output shows:
- Trading phase breakdown
- Decisions by phase
- Recent interactions with RSI analysis
- Price vs average comparisons

### 3. Generate Detailed Symbol Report

```bash
# Generate detailed report for specific symbol
python analyze_audit_trail.py --type report --symbol ETH/EUR --days 7
```

Output shows:
- All decisions for the symbol with timestamps
- Market data at each decision (RSI, price, averages)
- Decision reasons
- Position P&L information

## Example Analysis Scenarios

### Scenario 1: Why did the bot sell XRP at +26% profit?

```bash
# Get detailed report for XRP
python analyze_audit_trail.py --type report --symbol XRP/EUR --days 1

# Look for the specific SELL decision
# Check the LLM interaction file referenced in the decision
```

This will show:
- Market data when the decision was made (RSI 86.7 = overbought)
- The LLM's reasoning for selling
- Position age and P&L information

### Scenario 2: Why did the bot buy SHIB at a loss?

```bash
# Get detailed report for SHIB
python analyze_audit_trail.py --type report --symbol SHIB/EUR --days 1

# Check the LLM interaction for the BUY decision
```

This will show:
- Market data when SHIB was bought
- The LLM's reasoning for buying
- Whether it was an investment phase decision

### Scenario 3: Analyze bot performance over time

```bash
# Get decision breakdown for last 30 days
python analyze_audit_trail.py --type decisions --days 30

# Check portfolio snapshots for value changes
ls audit_trails/portfolio_snapshots/
```

## Manual File Analysis

### View Specific LLM Interaction

```bash
# Find the interaction file
ls audit_trails/llm_interactions/ | grep "ETH_EUR"

# View the content
cat audit_trails/llm_interactions/llm_interaction_ETH_EUR_20250712_161612_123.json | jq '.'
```

### View Specific Trading Decision

```bash
# Find the decision file
ls audit_trails/trading_decisions/ | grep "ETH_EUR_BUY"

# View the content
cat audit_trails/trading_decisions/trading_decision_ETH_EUR_BUY_20250712_161612_123.json | jq '.'
```

## Key Information in Audit Files

### Market Data Summary
- `current_price`: Current asset price
- `three_month_average`: 3-month average price
- `weekly_average`: Weekly average price
- `rsi`: Relative Strength Index
- `macd`: MACD indicator
- `volume_24h`: 24-hour volume

### Position Data
- `amount`: Crypto amount held
- `eur_value`: Current EUR value
- `buy_price`: Original buy price
- `profit_loss_pct`: Percentage P&L
- `profit_loss_eur`: EUR P&L

### Decision Context
- `trading_phase`: INVESTMENT or MANAGEMENT
- `decision_reason`: LLM's reasoning
- `execution_result`: Success/failure details

## Troubleshooting

### No Audit Files Found
- Check if the bot is running with audit trail enabled
- Verify the `audit_trails` directory exists
- Check file permissions

### Missing Information
- Some fields may be "Unknown" if data wasn't available
- Check the bot logs for any audit trail errors
- Verify the LLM client is properly integrated

### File Size Issues
- Audit files can grow large over time
- Use the cleanup function: `audit_trail.cleanup_old_audits(days_to_keep=30)`
- Consider archiving old files

## Integration with Trading Logic

The audit trail system is automatically integrated into:

1. **LLM Client**: Logs every interaction with the LLM
2. **Enhanced Multi-Bot**: Logs every trading decision and execution
3. **Portfolio Tracking**: Logs portfolio snapshots after each cycle

This provides complete transparency into why the bot made each decision and what information it had available. 