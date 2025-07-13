#!/usr/bin/env python3
"""
Performance Analysis CLI for Trading Bot
Command-line interface for analyzing trading performance and generating reports
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
from loguru import logger

from performance_monitor import PerformanceMonitor, performance_monitor

def setup_logging():
    """Setup logging for the CLI"""
    logger.add(
        "logs/performance_cli.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )

def cmd_summary(args):
    """Display performance summary"""
    print(f"\n🔍 Generating Performance Summary for last {args.days} days...")
    performance_monitor.print_performance_summary(args.days)

def cmd_report(args):
    """Generate detailed performance report"""
    print(f"\n📊 Generating Performance Report for last {args.days} days...")
    
    report = performance_monitor.generate_performance_report(args.days)
    
    if not report:
        print("❌ No data available for report generation")
        return
    
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE REPORT")
    print("="*80)
    
    summary = report.get('summary', {})
    print(f"Report Period:       {args.days} days")
    print(f"Report Date:         {report.get('report_date', 'Unknown')}")
    print(f"Total Return:        {summary.get('total_return_pct', 0):.2f}%")
    print(f"Annualized Return:   {summary.get('annualized_return_pct', 0):.2f}%")
    print(f"Win Rate:            {summary.get('win_rate_pct', 0):.1f}%")
    print(f"Max Drawdown:        {summary.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe Ratio:        {summary.get('sharpe_ratio', 0):.2f}")
    print(f"Profit Factor:       {summary.get('profit_factor', 0):.2f}")
    print(f"Total Trades:        {summary.get('total_trades', 0)}")
    print(f"Avg Win/Loss Ratio:  {summary.get('avg_win_loss_ratio', 0):.2f}")
    
    # Recent trades
    recent_trades = report.get('recent_trades', [])
    if recent_trades:
        print("\n" + "-"*80)
        print("RECENT TRADES (Last 10)")
        print("-"*80)
        for trade in recent_trades[:10]:
            profit_indicator = "📈" if trade.get('profit_loss', 0) > 0 else "📉" if trade.get('profit_loss', 0) < 0 else "➡️"
            print(f"{profit_indicator} {trade.get('timestamp', '')[:19]} | {trade.get('action', ''):4} | {trade.get('symbol', ''):10} | €{trade.get('value', 0):.2f} | P&L: €{trade.get('profit_loss', 0):+.2f}")
    
    # Portfolio evolution
    evolution = report.get('portfolio_evolution', [])
    if evolution:
        print("\n" + "-"*80)
        print("PORTFOLIO EVOLUTION")
        print("-"*80)
        for snapshot in evolution[-5:]:  # Last 5 snapshots
            print(f"{snapshot.get('timestamp', '')[:19]} | Value: €{snapshot.get('total_value', 0):.2f} | P&L: €{snapshot.get('total_pnl', 0):+.2f}")
    
    print("\n" + "="*80)
    
    if args.save:
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📄 Report saved to: {filename}")

def cmd_trades(args):
    """Display trading statistics"""
    print(f"\n📈 Trading Statistics for last {args.days} days...")
    
    stats = performance_monitor.get_trading_statistics(args.symbol, args.days)
    
    if not stats:
        print("❌ No trading data available")
        return
    
    print("\n" + "="*60)
    print("TRADING STATISTICS")
    print("="*60)
    
    # Group by symbol
    symbol_stats = {}
    for stat in stats:
        if isinstance(stat, dict):
            symbol = stat.get('symbol', '')
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'BUY': {}, 'SELL': {}, 'HOLD': {}}
            
            action = stat.get('action', '')
            symbol_stats[symbol][action] = stat
    
    for symbol, actions in symbol_stats.items():
        print(f"\n{symbol}:")
        print("-" * 20)
        
        for action, data in actions.items():
            if data:
                count = data.get('count', 0)
                avg_pnl = data.get('avg_pnl', 0)
                total_pnl = data.get('total_pnl', 0)
                avg_confidence = data.get('avg_confidence', 0)
                
                print(f"  {action:4}: {count:3} trades | Avg P&L: €{avg_pnl:+.2f} | Total: €{total_pnl:+.2f} | Confidence: {avg_confidence:.1f}%")

def cmd_portfolio(args):
    """Display portfolio history"""
    print(f"\n💼 Portfolio History for last {args.days} days...")
    
    history = performance_monitor.get_portfolio_history(args.days)
    
    if not history:
        print("❌ No portfolio data available")
        return
    
    print("\n" + "="*80)
    print("PORTFOLIO HISTORY")
    print("="*80)
    print(f"{'Date':<20} {'Total Value':<12} {'Cash':<12} {'Positions':<12} {'P&L':<12} {'Daily P&L':<12}")
    print("-" * 80)
    
    for snapshot in history[-20:]:  # Last 20 snapshots
        date = snapshot.get('timestamp', '')[:19]
        total_value = snapshot.get('total_value', 0)
        cash_balance = snapshot.get('cash_balance', 0)
        positions_value = snapshot.get('positions_value', 0)
        total_pnl = snapshot.get('total_pnl', 0)
        daily_pnl = snapshot.get('daily_pnl', 0)
        
        print(f"{date:<20} €{total_value:<11.2f} €{cash_balance:<11.2f} €{positions_value:<11.2f} €{total_pnl:<+11.2f} €{daily_pnl:<+11.2f}")

def cmd_chart(args):
    """Generate performance charts"""
    print(f"\n📊 Generating Performance Charts for last {args.days} days...")
    
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Get data
        history = performance_monitor.get_portfolio_history(args.days)
        
        if not history:
            print("❌ No data available for charting")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Trading Bot Performance - Last {args.days} Days', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(df['timestamp'], df['total_value'], linewidth=2, color='blue')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Value (€)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # P&L over time
        axes[0, 1].plot(df['timestamp'], df['total_pnl'], linewidth=2, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Profit & Loss Over Time')
        axes[0, 1].set_ylabel('P&L (€)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cash vs Positions
        axes[1, 0].plot(df['timestamp'], df['cash_balance'], label='Cash', linewidth=2)
        axes[1, 0].plot(df['timestamp'], df['positions_value'], label='Positions', linewidth=2)
        axes[1, 0].set_title('Cash vs Positions Value')
        axes[1, 0].set_ylabel('Value (€)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Daily P&L
        axes[1, 1].bar(df['timestamp'], df['daily_pnl'], alpha=0.7, 
                      color=['green' if x >= 0 else 'red' for x in df['daily_pnl']])
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Daily P&L')
        axes[1, 1].set_ylabel('Daily P&L (€)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"📈 Chart saved to: {chart_filename}")
        
        if args.show:
            plt.show()
        
    except ImportError:
        print("❌ Matplotlib and seaborn required for charting. Install with: pip install matplotlib seaborn")
    except Exception as e:
        logger.error(f"Error generating charts: {e}")
        print(f"❌ Error generating charts: {e}")

def cmd_compare(args):
    """Compare performance across different time periods"""
    print(f"\n🔍 Comparing Performance Across Time Periods...")
    
    periods = [7, 30, 90]  # 1 week, 1 month, 3 months
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Metric':<20} {'7 Days':<12} {'30 Days':<12} {'90 Days':<12}")
    print("-" * 80)
    
    metrics_data = {}
    for period in periods:
        metrics = performance_monitor.calculate_performance_metrics(period)
        metrics_data[period] = metrics
    
    # Display comparison
    metrics_to_compare = [
        ('Total Return (%)', 'total_return', 100),
        ('Annualized Return (%)', 'annualized_return', 100),
        ('Win Rate (%)', 'win_rate', 100),
        ('Sharpe Ratio', 'sharpe_ratio', 1),
        ('Max Drawdown (%)', 'max_drawdown', 100),
        ('Profit Factor', 'profit_factor', 1),
        ('Total Trades', 'total_trades', 1),
        ('Avg Win (€)', 'avg_win', 1),
        ('Avg Loss (€)', 'avg_loss', 1),
    ]
    
    for display_name, attr_name, multiplier in metrics_to_compare:
        values = []
        for period in periods:
            value = getattr(metrics_data[period], attr_name, 0) * multiplier
            if attr_name in ['total_trades']:
                values.append(f"{value:.0f}")
            else:
                values.append(f"{value:.2f}")
        
        print(f"{display_name:<20} {values[0]:<12} {values[1]:<12} {values[2]:<12}")

def cmd_export(args):
    """Export performance data to CSV"""
    print(f"\n📤 Exporting Performance Data...")
    
    try:
        # Export trades
        trades_stats = performance_monitor.get_trading_statistics(days=args.days)
        if trades_stats:
            trades_df = pd.DataFrame(trades_stats)
            trades_filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trades_df.to_csv(trades_filename, index=False)
            print(f"📊 Trades data exported to: {trades_filename}")
        
        # Export portfolio history
        portfolio_history = performance_monitor.get_portfolio_history(args.days)
        if portfolio_history:
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_filename = f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            portfolio_df.to_csv(portfolio_filename, index=False)
            print(f"💼 Portfolio data exported to: {portfolio_filename}")
        
        # Export performance metrics
        metrics = performance_monitor.calculate_performance_metrics(args.days)
        metrics_dict = {
            'metric': list(metrics.__dict__.keys()),
            'value': list(metrics.__dict__.values())
        }
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_filename = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        metrics_df.to_csv(metrics_filename, index=False)
        print(f"📈 Metrics data exported to: {metrics_filename}")
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        print(f"❌ Error exporting data: {e}")

def main():
    """Main CLI function"""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Trading Bot Performance Analysis CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Display performance summary')
    summary_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    summary_parser.set_defaults(func=cmd_summary)
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate detailed performance report')
    report_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    report_parser.add_argument('--save', action='store_true', help='Save report to file')
    report_parser.set_defaults(func=cmd_report)
    
    # Trades command
    trades_parser = subparsers.add_parser('trades', help='Display trading statistics')
    trades_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    trades_parser.add_argument('--symbol', type=str, help='Filter by specific symbol')
    trades_parser.set_defaults(func=cmd_trades)
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Display portfolio history')
    portfolio_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    portfolio_parser.set_defaults(func=cmd_portfolio)
    
    # Chart command
    chart_parser = subparsers.add_parser('chart', help='Generate performance charts')
    chart_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    chart_parser.add_argument('--show', action='store_true', help='Show charts interactively')
    chart_parser.set_defaults(func=cmd_chart)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare performance across time periods')
    compare_parser.set_defaults(func=cmd_compare)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export performance data to CSV')
    export_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    export_parser.set_defaults(func=cmd_export)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 