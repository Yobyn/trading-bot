#!/usr/bin/env python3
"""
Analyze Small Positions Script
Shows which cryptocurrency positions have value < €5
"""

from coinbase_smart_allocation_bot import CoinbaseSmartAllocationBot
from loguru import logger

def analyze_small_positions(threshold_eur: float = 5.0):
    """Analyze positions with value below threshold"""
    
    # Initialize the bot
    bot = CoinbaseSmartAllocationBot("coinbase_all_eur", "aggressive")
    
    print(f"\n🔍 ANALYZING POSITIONS WITH VALUE < €{threshold_eur:.2f}")
    print("=" * 60)
    
    # Get current positions
    existing_positions = bot.detect_existing_positions()
    
    if not existing_positions:
        print("No positions found")
        return
    
    # Categorize positions
    positions_to_sell = []
    positions_to_keep = []
    total_value_to_free = 0
    
    for symbol, position in existing_positions.items():
        value = position['eur_value']
        amount = position['amount']
        profit_loss_pct = position.get('profit_loss_pct', 0)
        
        if value < threshold_eur:
            positions_to_sell.append((symbol, position))
            total_value_to_free += value
        else:
            positions_to_keep.append((symbol, position))
    
    # Display results
    print(f"\n📊 CURRENT PORTFOLIO ANALYSIS:")
    print(f"Total positions: {len(existing_positions)}")
    print(f"Positions to sell (< €{threshold_eur:.2f}): {len(positions_to_sell)}")
    print(f"Positions to keep (≥ €{threshold_eur:.2f}): {len(positions_to_keep)}")
    
    if positions_to_sell:
        print(f"\n🎯 POSITIONS TO SELL (< €{threshold_eur:.2f}):")
        print("-" * 50)
        for symbol, position in positions_to_sell:
            amount = position['amount']
            value = position['eur_value']
            profit_loss_pct = position.get('profit_loss_pct', 0)
            profit_indicator = "📈" if profit_loss_pct > 0 else "📉" if profit_loss_pct < 0 else "➡️"
            print(f"  • {symbol:10s}: {amount:>12.6f} tokens = €{value:>6.2f} {profit_indicator} {profit_loss_pct:+.1f}%")
        
        print(f"\n💰 TOTAL VALUE TO FREE UP: €{total_value_to_free:.2f}")
    
    if positions_to_keep:
        print(f"\n✅ POSITIONS TO KEEP (≥ €{threshold_eur:.2f}):")
        print("-" * 50)
        for symbol, position in positions_to_keep:
            amount = position['amount']
            value = position['eur_value']
            profit_loss_pct = position.get('profit_loss_pct', 0)
            profit_indicator = "📈" if profit_loss_pct > 0 else "📉" if profit_loss_pct < 0 else "➡️"
            print(f"  • {symbol:10s}: {amount:>12.6f} tokens = €{value:>6.2f} {profit_indicator} {profit_loss_pct:+.1f}%")
    
    print(f"\n🚀 IMPACT ON INVESTMENT PHASE:")
    if total_value_to_free > 10:
        print(f"✅ Freeing up €{total_value_to_free:.2f} will trigger INVESTMENT PHASE (>€10)")
        print("   The bot will ask LLM to pick the SINGLE BEST crypto from all 41 options")
        print("   and invest ALL the freed cash into that one crypto!")
    else:
        print(f"⚠️  Freeing up €{total_value_to_free:.2f} may not trigger INVESTMENT PHASE (need >€10)")
    
    print("\n" + "=" * 60)
    
    if positions_to_sell:
        print(f"\nTo execute the sales, run:")
        print(f"python sell_small_positions.py")

if __name__ == "__main__":
    analyze_small_positions(threshold_eur=5.0) 