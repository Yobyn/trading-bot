#!/usr/bin/env python3
"""
Risk Management CLI for Trading Bot
Command-line interface for risk analysis and monitoring
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
from loguru import logger

from risk_manager import RiskManager, risk_manager

def setup_logging():
    """Setup logging for the CLI"""
    logger.add(
        "logs/risk_cli.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )

def cmd_summary(args):
    """Display risk summary"""
    print("\n🔍 Risk Management Summary")
    risk_manager.print_risk_report()

def cmd_alerts(args):
    """Display recent risk alerts"""
    print(f"\n🚨 Risk Alerts for last {args.days} days...")
    
    summary = risk_manager.get_risk_summary()
    alerts = summary.get('recent_alerts', [])
    
    if not alerts:
        print("✅ No risk alerts found")
        return
    
    print("\n" + "="*80)
    print("RISK ALERTS")
    print("="*80)
    
    # Group alerts by severity
    severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    severity_icons = {
        'CRITICAL': '🚨',
        'HIGH': '🔴',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    
    for severity in severity_order:
        severity_alerts = [a for a in alerts if a.get('severity') == severity]
        if severity_alerts:
            print(f"\n{severity_icons[severity]} {severity} ALERTS ({len(severity_alerts)})")
            print("-" * 40)
            
            for alert in severity_alerts:
                timestamp = alert.get('timestamp', '')[:19]
                alert_type = alert.get('alert_type', '')
                symbol = alert.get('symbol', 'PORTFOLIO')
                message = alert.get('message', '')
                recommendation = alert.get('recommendation', '')
                
                print(f"  {timestamp} | {symbol} | {alert_type}")
                print(f"    {message}")
                print(f"    💡 {recommendation}")
                print()

def cmd_positions(args):
    """Display position risk analysis"""
    print(f"\n📊 Position Risk Analysis...")
    
    summary = risk_manager.get_risk_summary()
    position_risks = summary.get('position_risks', [])
    
    if not position_risks:
        print("❌ No position risk data available")
        return
    
    print("\n" + "="*100)
    print("POSITION RISK ANALYSIS")
    print("="*100)
    print(f"{'Symbol':<12} {'Size (€)':<12} {'VaR (€)':<12} {'Beta':<8} {'Correlation':<12} {'Concentration':<14} {'Risk Score':<12}")
    print("-" * 100)
    
    # Group by symbol and get latest data
    symbol_data = {}
    for pos in position_risks:
        symbol = pos.get('symbol', '')
        if symbol not in symbol_data or pos.get('timestamp', '') > symbol_data[symbol].get('timestamp', ''):
            symbol_data[symbol] = pos
    
    # Sort by risk contribution
    sorted_positions = sorted(symbol_data.values(), 
                            key=lambda x: x.get('risk_contribution', 0), 
                            reverse=True)
    
    for pos in sorted_positions:
        symbol = pos.get('symbol', '')
        size = pos.get('position_size', 0)
        var = pos.get('value_at_risk', 0)
        beta = pos.get('beta', 1.0)
        correlation = pos.get('correlation_with_portfolio', 0)
        concentration = pos.get('concentration_pct', 0)
        risk_score = pos.get('risk_contribution', 0)
        
        # Risk indicators
        risk_indicator = "🔴" if concentration > 0.25 else "🟡" if concentration > 0.15 else "🟢"
        
        print(f"{risk_indicator} {symbol:<10} €{size:<11.2f} €{var:<11.2f} {beta:<7.2f} {correlation:<11.2f} {concentration:<13.1%} {risk_score:<11.2f}")

def cmd_optimize(args):
    """Display portfolio optimization suggestions"""
    print(f"\n🎯 Portfolio Optimization Analysis...")
    
    # This would require price history data
    print("⚠️  Portfolio optimization requires historical price data")
    print("💡 This feature will be enhanced when integrated with the trading bot")
    
    # Show current risk thresholds
    summary = risk_manager.get_risk_summary()
    thresholds = summary.get('risk_thresholds', {})
    
    print("\n" + "="*60)
    print("RISK THRESHOLDS")
    print("="*60)
    
    for key, value in thresholds.items():
        display_name = key.replace('_', ' ').title()
        if 'pct' in key or 'ratio' in key:
            print(f"{display_name:<30}: {value:.1%}")
        else:
            print(f"{display_name:<30}: {value:.2f}")

def cmd_correlation(args):
    """Display correlation analysis"""
    print(f"\n🔗 Correlation Analysis...")
    
    summary = risk_manager.get_risk_summary()
    metrics = summary.get('latest_metrics', {})
    
    if not metrics:
        print("❌ No correlation data available")
        return
    
    correlation_risk = metrics.get('correlation_risk', 0)
    concentration_risk = metrics.get('concentration_risk', 0)
    
    print("\n" + "="*60)
    print("CORRELATION & CONCENTRATION ANALYSIS")
    print("="*60)
    print(f"Portfolio Correlation Risk:    {correlation_risk:.2f}")
    print(f"Portfolio Concentration Risk:  {concentration_risk:.2f}")
    
    # Risk assessment
    if correlation_risk > 0.7:
        print("🔴 HIGH correlation risk - positions are highly correlated")
        print("💡 Consider diversifying into uncorrelated assets")
    elif correlation_risk > 0.5:
        print("🟡 MEDIUM correlation risk - some correlation present")
        print("💡 Monitor correlation levels and consider rebalancing")
    else:
        print("🟢 LOW correlation risk - good diversification")
    
    if concentration_risk > 0.5:
        print("🔴 HIGH concentration risk - portfolio is concentrated")
        print("💡 Consider reducing large positions")
    elif concentration_risk > 0.3:
        print("🟡 MEDIUM concentration risk - some concentration present")
        print("💡 Monitor position sizes")
    else:
        print("🟢 LOW concentration risk - well-balanced portfolio")

def cmd_var(args):
    """Display Value at Risk analysis"""
    print(f"\n📉 Value at Risk (VaR) Analysis...")
    
    summary = risk_manager.get_risk_summary()
    metrics = summary.get('latest_metrics', {})
    
    if not metrics:
        print("❌ No VaR data available")
        return
    
    portfolio_var = metrics.get('portfolio_var', 0)
    portfolio_cvar = metrics.get('portfolio_cvar', 0)
    max_drawdown = metrics.get('max_drawdown', 0)
    volatility = metrics.get('volatility', 0)
    
    print("\n" + "="*60)
    print("VALUE AT RISK ANALYSIS")
    print("="*60)
    print(f"Portfolio VaR (95%):          {portfolio_var:.2%}")
    print(f"Portfolio CVaR (95%):         {portfolio_cvar:.2%}")
    print(f"Maximum Drawdown:             {max_drawdown:.2%}")
    print(f"Portfolio Volatility:         {volatility:.2%}")
    
    # Risk assessment
    threshold_var = 0.05  # 5%
    if abs(portfolio_var) > threshold_var:
        print(f"🔴 VaR exceeds threshold ({threshold_var:.1%})")
        print("💡 Consider reducing position sizes or increasing diversification")
    else:
        print(f"🟢 VaR within acceptable range (<{threshold_var:.1%})")
    
    if abs(max_drawdown) > 0.15:  # 15%
        print("🔴 High maximum drawdown detected")
        print("💡 Review stop-loss settings and risk management")
    else:
        print("🟢 Maximum drawdown within acceptable range")

def cmd_thresholds(args):
    """Display and manage risk thresholds"""
    print(f"\n⚙️  Risk Threshold Management...")
    
    summary = risk_manager.get_risk_summary()
    thresholds = summary.get('risk_thresholds', {})
    
    print("\n" + "="*80)
    print("CURRENT RISK THRESHOLDS")
    print("="*80)
    print(f"{'Threshold':<35} {'Current Value':<15} {'Status':<10}")
    print("-" * 80)
    
    metrics = summary.get('latest_metrics', {})
    
    threshold_mapping = {
        'max_portfolio_var': ('portfolio_var', 'Portfolio VaR'),
        'max_position_concentration': ('concentration_risk', 'Position Concentration'),
        'max_correlation': ('correlation_risk', 'Correlation Risk'),
        'max_drawdown': ('max_drawdown', 'Maximum Drawdown'),
        'min_sharpe_ratio': ('sharpe_ratio', 'Sharpe Ratio'),
        'max_beta': ('beta', 'Portfolio Beta'),
        'max_volatility': ('volatility', 'Portfolio Volatility')
    }
    
    for threshold_key, (metric_key, display_name) in threshold_mapping.items():
        threshold_value = thresholds.get(threshold_key, 0)
        current_value = metrics.get(metric_key, 0)
        
        # Determine status
        if 'min_' in threshold_key:
            status = "🟢 OK" if current_value >= threshold_value else "🔴 BREACH"
        else:
            status = "🟢 OK" if abs(current_value) <= threshold_value else "🔴 BREACH"
        
        if 'ratio' in threshold_key or 'pct' in threshold_key or 'var' in threshold_key:
            print(f"{display_name:<35} {current_value:<14.2%} {status}")
        else:
            print(f"{display_name:<35} {current_value:<14.2f} {status}")

def cmd_export(args):
    """Export risk data to CSV"""
    print(f"\n📤 Exporting Risk Data...")
    
    try:
        summary = risk_manager.get_risk_summary()
        
        # Export risk metrics
        metrics = summary.get('latest_metrics', {})
        if metrics:
            metrics_df = pd.DataFrame([metrics])
            metrics_filename = f"risk_metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            metrics_df.to_csv(metrics_filename, index=False)
            print(f"📊 Risk metrics exported to: {metrics_filename}")
        
        # Export position risks
        position_risks = summary.get('position_risks', [])
        if position_risks:
            positions_df = pd.DataFrame(position_risks)
            positions_filename = f"position_risks_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            positions_df.to_csv(positions_filename, index=False)
            print(f"📈 Position risks exported to: {positions_filename}")
        
        # Export alerts
        alerts = summary.get('recent_alerts', [])
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_filename = f"risk_alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            alerts_df.to_csv(alerts_filename, index=False)
            print(f"🚨 Risk alerts exported to: {alerts_filename}")
        
        if not any([metrics, position_risks, alerts]):
            print("❌ No risk data available for export")
            
    except Exception as e:
        logger.error(f"Error exporting risk data: {e}")
        print(f"❌ Error exporting data: {e}")

def cmd_monitor(args):
    """Start risk monitoring mode"""
    print(f"\n👁️  Risk Monitoring Mode...")
    print("This would start a continuous monitoring process")
    print("💡 Integration with trading bot required for real-time monitoring")
    
    # Show what would be monitored
    print("\n" + "="*60)
    print("MONITORING CHECKLIST")
    print("="*60)
    print("✓ Portfolio Value at Risk (VaR)")
    print("✓ Position Concentration Limits")
    print("✓ Correlation Risk Levels")
    print("✓ Maximum Drawdown Limits")
    print("✓ Volatility Thresholds")
    print("✓ Liquidity Risk Scores")
    print("✓ Beta Exposure Limits")
    print("✓ Sharpe Ratio Minimums")
    
    print("\n💡 Use 'risk_cli.py alerts' to check for current violations")

def main():
    """Main CLI function"""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Trading Bot Risk Management CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Display risk summary')
    summary_parser.set_defaults(func=cmd_summary)
    
    # Alerts command
    alerts_parser = subparsers.add_parser('alerts', help='Display risk alerts')
    alerts_parser.add_argument('--days', type=int, default=7, help='Number of days to check (default: 7)')
    alerts_parser.set_defaults(func=cmd_alerts)
    
    # Positions command
    positions_parser = subparsers.add_parser('positions', help='Display position risk analysis')
    positions_parser.set_defaults(func=cmd_positions)
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Display portfolio optimization suggestions')
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # Correlation command
    correlation_parser = subparsers.add_parser('correlation', help='Display correlation analysis')
    correlation_parser.set_defaults(func=cmd_correlation)
    
    # VaR command
    var_parser = subparsers.add_parser('var', help='Display Value at Risk analysis')
    var_parser.set_defaults(func=cmd_var)
    
    # Thresholds command
    thresholds_parser = subparsers.add_parser('thresholds', help='Display risk thresholds')
    thresholds_parser.set_defaults(func=cmd_thresholds)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export risk data to CSV')
    export_parser.set_defaults(func=cmd_export)
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start risk monitoring mode')
    monitor_parser.set_defaults(func=cmd_monitor)
    
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