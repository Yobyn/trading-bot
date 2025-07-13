#!/usr/bin/env python3
"""
Audit Trail Analysis Utility
Analyze trading decisions and LLM interactions from audit trails
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from audit_trail import AuditTrail
from loguru import logger

def analyze_trading_decisions(days_back: int = 7, symbol: Optional[str] = None):
    """Analyze trading decisions from the last N days"""
    audit_trail = AuditTrail()
    
    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    logger.info(f"📊 Analyzing trading decisions from {start_date} to {end_date}")
    
    # Get recent decisions
    decisions = audit_trail.get_recent_decisions(symbol=symbol, limit=1000)
    
    if not decisions:
        logger.warning("No trading decisions found in the specified date range")
        return
    
    # Filter by date range
    filtered_decisions = []
    for decision in decisions:
        decision_date = decision.get('timestamp', '')[:10]  # YYYY-MM-DD
        if start_date <= decision_date <= end_date:
            filtered_decisions.append(decision)
    
    if not filtered_decisions:
        logger.warning(f"No trading decisions found between {start_date} and {end_date}")
        return
    
    # Analyze decisions
    actions = [d.get('action') for d in filtered_decisions]
    symbols = [d.get('symbol') for d in filtered_decisions]
    
    # Count actions
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    # Count symbols
    symbol_counts = {}
    for symbol in symbols:
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    
    # Print summary
    logger.info(f"📈 Trading Decision Summary ({len(filtered_decisions)} decisions):")
    logger.info(f"  Date Range: {start_date} to {end_date}")
    logger.info(f"  Total Decisions: {len(filtered_decisions)}")
    
    logger.info(f"  Action Breakdown:")
    for action, count in sorted(action_counts.items()):
        percentage = (count / len(filtered_decisions)) * 100
        logger.info(f"    {action}: {count} ({percentage:.1f}%)")
    
    logger.info(f"  Most Traded Symbols:")
    for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"    {symbol}: {count} decisions")
    
    # Analyze specific decisions
    logger.info(f"\n🔍 Detailed Decision Analysis:")
    
    # Show recent BUY decisions
    buy_decisions = [d for d in filtered_decisions if d.get('action') == 'BUY']
    if buy_decisions:
        logger.info(f"  Recent BUY Decisions ({len(buy_decisions)}):")
        for decision in buy_decisions[-5:]:  # Last 5
            symbol = decision.get('symbol')
            timestamp = decision.get('timestamp', '')[:19]  # YYYY-MM-DD HH:MM:SS
            reason = decision.get('decision_reason', 'No reason provided')[:100] + "..."
            market_data = decision.get('market_data_at_decision', {})
            rsi = market_data.get('rsi', 'Unknown')
            current_price = market_data.get('current_price', 'Unknown')
            
            logger.info(f"    {timestamp} | {symbol} | RSI: {rsi} | Price: €{current_price}")
            logger.info(f"      Reason: {reason}")
    
    # Show recent SELL decisions
    sell_decisions = [d for d in filtered_decisions if d.get('action') == 'SELL']
    if sell_decisions:
        logger.info(f"  Recent SELL Decisions ({len(sell_decisions)}):")
        for decision in sell_decisions[-5:]:  # Last 5
            symbol = decision.get('symbol')
            timestamp = decision.get('timestamp', '')[:19]
            reason = decision.get('decision_reason', 'No reason provided')[:100] + "..."
            market_data = decision.get('market_data_at_decision', {})
            rsi = market_data.get('rsi', 'Unknown')
            current_price = market_data.get('current_price', 'Unknown')
            position_data = decision.get('position_data', {})
            profit_loss = position_data.get('profit_loss_pct', 'Unknown')
            
            logger.info(f"    {timestamp} | {symbol} | RSI: {rsi} | Price: €{current_price} | P&L: {profit_loss}%")
            logger.info(f"      Reason: {reason}")

def analyze_llm_interactions(days_back: int = 7, symbol: Optional[str] = None):
    """Analyze LLM interactions from the last N days"""
    audit_trail = AuditTrail()
    
    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    logger.info(f"🤖 Analyzing LLM interactions from {start_date} to {end_date}")
    
    # Get LLM interactions
    interactions = []
    
    # Check if directory exists
    if not os.path.exists(audit_trail.llm_interactions_dir):
        logger.info(f"LLM interactions directory does not exist: {audit_trail.llm_interactions_dir}")
        return
    
    for filename in os.listdir(audit_trail.llm_interactions_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(audit_trail.llm_interactions_dir, filename)
        try:
            with open(filepath, 'r') as f:
                interaction = json.load(f)
            
            # Apply filters
            if symbol and interaction.get('symbol') != symbol:
                continue
            
            interaction_date = interaction.get('timestamp', '')[:10]  # YYYY-MM-DD
            if start_date <= interaction_date <= end_date:
                interactions.append(interaction)
        except Exception as e:
            logger.warning(f"Could not load interaction file {filename}: {e}")
    
    if not interactions:
        logger.warning("No LLM interactions found in the specified date range")
        return
    
    # Sort by timestamp
    interactions.sort(key=lambda x: x.get('timestamp', ''))
    
    logger.info(f"📊 LLM Interaction Summary ({len(interactions)} interactions):")
    
    # Analyze decisions by trading phase
    phases = {}
    decisions_by_phase = {}
    
    for interaction in interactions:
        phase = interaction.get('trading_phase', 'Unknown')
        decision = interaction.get('parsed_decision', {})
        action = decision.get('action', 'Unknown')
        
        phases[phase] = phases.get(phase, 0) + 1
        
        if phase not in decisions_by_phase:
            decisions_by_phase[phase] = {}
        decisions_by_phase[phase][action] = decisions_by_phase[phase].get(action, 0) + 1
    
    logger.info(f"  Trading Phases:")
    for phase, count in phases.items():
        logger.info(f"    {phase}: {count} interactions")
    
    logger.info(f"  Decisions by Phase:")
    for phase, decisions in decisions_by_phase.items():
        logger.info(f"    {phase}:")
        for action, count in decisions.items():
            logger.info(f"      {action}: {count}")
    
    # Show recent interactions with RSI analysis
    logger.info(f"\n🔍 Recent LLM Interactions with RSI Analysis:")
    
    for interaction in interactions[-10:]:  # Last 10
        symbol = interaction.get('symbol')
        timestamp = interaction.get('timestamp', '')[:19]
        decision = interaction.get('parsed_decision', {})
        action = decision.get('action', 'Unknown')
        reason = decision.get('reason', 'No reason')[:80] + "..."
        market_data = interaction.get('market_data_summary', {})
        rsi = market_data.get('rsi', 'Unknown')
        current_price = market_data.get('current_price', 'Unknown')
        three_month_avg = market_data.get('three_month_average', 'Unknown')
        
        # Calculate price vs average
        price_vs_avg = "Unknown"
        if isinstance(current_price, (int, float)) and isinstance(three_month_avg, (int, float)) and three_month_avg > 0:
            diff_pct = ((current_price - three_month_avg) / three_month_avg) * 100
            if diff_pct > 0:
                price_vs_avg = f"+{diff_pct:.1f}% above avg"
            else:
                price_vs_avg = f"{diff_pct:.1f}% below avg"
        
        logger.info(f"    {timestamp} | {symbol} | {action} | RSI: {rsi} | {price_vs_avg}")
        logger.info(f"      Reason: {reason}")

def generate_decision_report(symbol: str, days_back: int = 7):
    """Generate a detailed report for a specific symbol"""
    audit_trail = AuditTrail()
    
    logger.info(f"📋 Generating detailed report for {symbol} (last {days_back} days)")
    
    # Get decisions for this symbol
    decisions = audit_trail.get_recent_decisions(symbol=symbol, limit=1000)
    
    if not decisions:
        logger.warning(f"No decisions found for {symbol}")
        return
    
    # Filter by date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    filtered_decisions = []
    for decision in decisions:
        decision_time = datetime.fromisoformat(decision.get('timestamp', '').replace('Z', '+00:00'))
        if start_date <= decision_time <= end_date:
            filtered_decisions.append(decision)
    
    if not filtered_decisions:
        logger.warning(f"No decisions found for {symbol} in the last {days_back} days")
        return
    
    # Sort by timestamp
    filtered_decisions.sort(key=lambda x: x.get('timestamp', ''))
    
    logger.info(f"📊 {symbol} Decision Report ({len(filtered_decisions)} decisions):")
    
    # Show all decisions with market data
    for decision in filtered_decisions:
        timestamp = decision.get('timestamp', '')[:19]
        action = decision.get('action', 'Unknown')
        reason = decision.get('decision_reason', 'No reason provided')
        market_data = decision.get('market_data_at_decision', {})
        
        rsi = market_data.get('rsi', 'Unknown')
        current_price = market_data.get('current_price', 'Unknown')
        three_month_avg = market_data.get('three_month_average', 'Unknown')
        
        # Calculate price vs average
        price_vs_avg = "Unknown"
        if isinstance(current_price, (int, float)) and isinstance(three_month_avg, (int, float)) and three_month_avg > 0:
            diff_pct = ((current_price - three_month_avg) / three_month_avg) * 100
            if diff_pct > 0:
                price_vs_avg = f"+{diff_pct:.1f}% above avg"
            else:
                price_vs_avg = f"{diff_pct:.1f}% below avg"
        
        # Get position data
        position_data = decision.get('position_data', {})
        profit_loss = position_data.get('profit_loss_pct', 'Unknown') if position_data else 'N/A'
        
        logger.info(f"  {timestamp} | {action} | RSI: {rsi} | Price: €{current_price} | {price_vs_avg} | P&L: {profit_loss}%")
        logger.info(f"    Reason: {reason[:150]}...")
        logger.info("")

def main():
    """Main function to run audit trail analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trading bot audit trails')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze (default: 7)')
    parser.add_argument('--symbol', type=str, help='Specific symbol to analyze')
    parser.add_argument('--type', choices=['decisions', 'interactions', 'report'], default='decisions', 
                       help='Type of analysis to perform')
    
    args = parser.parse_args()
    
    if args.type == 'decisions':
        analyze_trading_decisions(days_back=args.days, symbol=args.symbol)
    elif args.type == 'interactions':
        analyze_llm_interactions(days_back=args.days, symbol=args.symbol)
    elif args.type == 'report':
        if not args.symbol:
            logger.error("Symbol is required for detailed report")
            return
        generate_decision_report(symbol=args.symbol, days_back=args.days)

if __name__ == "__main__":
    main() 