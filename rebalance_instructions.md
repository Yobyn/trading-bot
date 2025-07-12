# Rebalancing Options

## Option 1: Manual Rebalancing Script (Recommended)
```bash
python manual_rebalance.py
```

## Option 2: Enhanced Bot with Force Liquidation
Stop current bot and temporarily increase cash threshold:

```python
# In enhanced_multi_bot.py, temporarily change line ~523:
if (investable_cash > 5 and len(meaningful_positions) == 0) or (investable_cash > 10):
```

## Option 3: Regular Coinbase Bot Rebalancing
```bash
python -c "
import asyncio
from coinbase_smart_allocation_bot import CoinbaseSmartAllocationBot

async def run():
    bot = CoinbaseSmartAllocationBot('coinbase_all_eur', 'aggressive')
    await bot.run_rebalancing_cycle()

asyncio.run(run())
"
```

## Option 4: CLI Command
```bash
python coinbase_cli.py --rebalance --portfolio coinbase_all_eur --strategy aggressive
```
