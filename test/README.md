# Trading Bot Tests

This directory contains all test files for the trading bot project.

## Test Files

- **`test_bot.py`** - General bot functionality tests
- **`test_coinbase_advanced.py`** - Coinbase Advanced API integration tests
- **`test_coinbase_basic.py`** - Basic Coinbase API functionality tests
- **`test_coinbase_connection.py`** - Coinbase connection and authentication tests
- **`test_coinbase.py`** - Legacy Coinbase API tests

## Running Tests

### Run all tests:
```bash
cd test
python -m pytest *.py -v
```

### Run specific test file:
```bash
cd test
python test_coinbase_advanced.py
```

### Run from project root:
```bash
python -m test.test_coinbase_advanced
```

## Test Requirements

Make sure you have the following environment variables set in your `.env` file:
- `COINBASE_API_KEY` - Your Coinbase Advanced API key
- `COINBASE_API_SECRET_FILE` - Path to your private key file

## Test Categories

### Connection Tests
- API authentication
- Network connectivity
- Rate limiting

### Functionality Tests
- Market data fetching
- Order placement
- Account balance retrieval

### Integration Tests
- End-to-end trading workflows
- Error handling
- Data validation

## Notes

- Tests are designed to work with both live and paper trading modes
- Some tests may require actual API credentials
- Always test with small amounts in paper trading mode first 