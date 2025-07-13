#!/usr/bin/env python3
"""
Web Dashboard for Trading Bot
Real-time monitoring and control interface with performance analytics
"""

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import time
from loguru import logger

# Import our trading bot components
from enhanced_multi_bot import EnhancedMultiAssetBot
from performance_monitor import performance_monitor, PerformanceMonitor
from risk_manager import risk_manager, RiskManager
from audit_trail import audit_trail, AuditTrail
from config import config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading-bot-dashboard-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global bot instance
bot_instance = None
bot_thread = None
bot_status = {
    'running': False,
    'last_update': None,
    'error': None,
    'portfolio_value': 0,
    'total_pnl': 0,
    'position_count': 0
}

class DashboardDataProvider:
    """Provides data for the dashboard"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.risk_manager = RiskManager()
        self.audit_trail = AuditTrail()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        try:
            if bot_instance:
                status = bot_instance.get_status()
                portfolio = status.get('portfolio', {})
                
                # Get positions and filter out those with value < €1
                existing_positions = bot_instance.coinbase_bot.detect_existing_positions()
                tradeable_positions = {k: v for k, v in existing_positions.items() if v['eur_value'] >= 1.0}
                
                # Recalculate values based on tradeable positions only
                tradeable_total_value = sum(pos['eur_value'] for pos in tradeable_positions.values())
                tradeable_total_pnl = sum(pos.get('profit_loss_eur', 0) for pos in tradeable_positions.values())
                
                return {
                    'total_value': portfolio.get('portfolio_value', 0),  # Keep total portfolio value
                    'total_pnl': tradeable_total_pnl,  # Only show P&L from tradeable positions
                    'position_count': len(tradeable_positions),  # Only count tradeable positions
                    'last_update': datetime.now().isoformat(),
                    'is_running': status.get('is_running', False),
                    'portfolio_name': status.get('portfolio_name', 'Unknown'),
                    'strategy_name': status.get('strategy_name', 'Unknown')
                }
            else:
                return {
                    'total_value': 0,
                    'total_pnl': 0,
                    'position_count': 0,
                    'last_update': None,
                    'is_running': False,
                    'portfolio_name': 'Not Started',
                    'strategy_name': 'Not Started'
                }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {
                'total_value': 0,
                'total_pnl': 0,
                'position_count': 0,
                'last_update': None,
                'is_running': False,
                'error': str(e)
            }
    
    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for the last N days"""
        try:
            metrics = self.performance_monitor.calculate_performance_metrics(days)
            return {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'volatility': metrics.volatility
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        try:
            # Get positions data for risk calculation - only include positions ≥ €1
            positions = {}
            price_history = {}
            
            if bot_instance:
                existing_positions = bot_instance.coinbase_bot.detect_existing_positions()
                for symbol, pos in existing_positions.items():
                    if pos['eur_value'] >= 1.0:  # Only include tradeable positions
                        positions[symbol] = pos['eur_value']
                        # For now, use empty price history - would need to implement historical data
                        price_history[symbol] = []
            
            risk_metrics = self.risk_manager.calculate_portfolio_risk(positions, price_history)
            return {
                'portfolio_var': risk_metrics.portfolio_var,
                'portfolio_cvar': risk_metrics.portfolio_cvar,
                'concentration_risk': risk_metrics.concentration_risk,
                'correlation_risk': risk_metrics.correlation_risk,
                'volatility': risk_metrics.volatility
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades"""
        try:
            trades = self.performance_monitor.get_recent_trades(limit)
            return [
                {
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'action': trade.action,
                    'price': trade.price,
                    'quantity': trade.quantity,
                    'value': trade.value,
                    'profit_loss': trade.profit_loss,
                    'profit_loss_pct': trade.profit_loss_pct,
                    'reason': trade.reason,
                    'confidence': trade.confidence
                }
                for trade in trades
            ]
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def get_portfolio_chart_data(self, days: int = 30) -> Dict[str, Any]:
        """Get portfolio value chart data"""
        try:
            snapshots = self.performance_monitor.get_portfolio_snapshots(days)
            
            timestamps = [snapshot.timestamp for snapshot in snapshots]
            values = [snapshot.total_value for snapshot in snapshots]
            pnl = [snapshot.total_pnl for snapshot in snapshots]
            
            return {
                'timestamps': timestamps,
                'values': values,
                'pnl': pnl
            }
        except Exception as e:
            logger.error(f"Error getting portfolio chart data: {e}")
            return {'timestamps': [], 'values': [], 'pnl': []}
    
    def get_positions_data(self) -> List[Dict[str, Any]]:
        """Get current positions data - filters out positions with value < €1"""
        try:
            if bot_instance:
                existing_positions = bot_instance.coinbase_bot.detect_existing_positions()
                # Filter out positions with value less than €1
                return [
                    {
                        'symbol': symbol,
                        'amount': pos['amount'],
                        'eur_value': pos['eur_value'],
                        'buy_price': pos.get('buy_price', 0),
                        'profit_loss_pct': pos.get('profit_loss_pct', 0),
                        'profit_loss_eur': pos.get('profit_loss_eur', 0)
                    }
                    for symbol, pos in existing_positions.items()
                    if pos['eur_value'] >= 1.0  # Only show positions worth €1 or more
                ]
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting positions data: {e}")
            return []
    
    def get_recent_decisions(self, limit: int = 10, days: Optional[int] = None, action: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent trading decisions from audit trail with filtering"""
        try:
            # Use the new database method for better performance
            days_filter = days if days is not None else 30
            decisions = self.audit_trail.get_recent_decisions_db(
                symbol=symbol if symbol and symbol != 'all' else None,
                action=action if action and action != 'all' else None,
                days=days_filter,
                limit=limit
            )
            
            # If no database results, fallback to JSON files
            if not decisions:
                decisions = self.audit_trail.get_recent_decisions(limit=limit)
                
                # Filter by days if specified
                if days is not None:
                    from datetime import datetime, timedelta
                    cutoff_date = datetime.now() - timedelta(days=days)
                    decisions = [d for d in decisions if datetime.fromisoformat(d.get('timestamp', '').replace('Z', '+00:00')) >= cutoff_date]
                
                # Filter by action if specified
                if action and action != 'all':
                    decisions = [d for d in decisions if d.get('action') == action]
                
                # Filter by symbol if specified
                if symbol and symbol != 'all':
                    decisions = [d for d in decisions if d.get('symbol') == symbol]
            
            # Format the results consistently
            formatted_decisions = []
            for decision in decisions:
                # Handle both database format and JSON format
                if isinstance(decision, dict):
                    # Database format
                    if 'current_price' in decision:
                        formatted_decisions.append({
                            'id': f"{decision.get('timestamp', '')}_{decision.get('symbol', '')}",
                            'timestamp': decision.get('timestamp'),
                            'symbol': decision.get('symbol'),
                            'action': decision.get('action'),
                            'reason': decision.get('decision_reason', ''),
                            'market_price': decision.get('current_price'),
                            'rsi': decision.get('rsi'),
                            'macd': decision.get('macd'),
                            'volume': decision.get('volume_24h'),
                            'three_month_avg': decision.get('three_month_avg'),
                            'confidence': decision.get('confidence'),
                            'success': decision.get('execution_success', False),
                            'execution_result': decision.get('execution_result', {})
                        })
                    # JSON format (fallback)
                    else:
                        formatted_decisions.append({
                            'id': f"{decision.get('timestamp', '')}_{decision.get('symbol', '')}",
                            'timestamp': decision.get('timestamp'),
                            'symbol': decision.get('symbol'),
                            'action': decision.get('action'),
                            'reason': decision.get('decision_reason', ''),
                            'market_price': decision.get('market_data_at_decision', {}).get('current_price'),
                            'rsi': decision.get('market_data_at_decision', {}).get('rsi'),
                            'macd': decision.get('market_data_at_decision', {}).get('macd'),
                            'volume': decision.get('market_data_at_decision', {}).get('volume_24h'),
                            'three_month_avg': decision.get('market_data_at_decision', {}).get('three_month_average'),
                            'confidence': decision.get('market_data_at_decision', {}).get('confidence'),
                            'success': decision.get('execution_result', {}).get('success', False),
                            'execution_result': decision.get('execution_result', {})
                        })
            
            return formatted_decisions
        except Exception as e:
            logger.error(f"Error getting recent decisions: {e}")
            return []

# Initialize data provider
data_provider = DashboardDataProvider()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/portfolio')
def api_portfolio():
    """API endpoint for portfolio data"""
    return jsonify(data_provider.get_portfolio_summary())

@app.route('/api/performance')
def api_performance():
    """API endpoint for performance metrics"""
    days = request.args.get('days', 30, type=int)
    return jsonify(data_provider.get_performance_metrics(days))

@app.route('/api/risk')
def api_risk():
    """API endpoint for risk metrics"""
    return jsonify(data_provider.get_risk_metrics())

@app.route('/api/trades')
def api_trades():
    """API endpoint for recent trades"""
    limit = request.args.get('limit', 20, type=int)
    return jsonify(data_provider.get_recent_trades(limit))

@app.route('/api/chart')
def api_chart():
    """API endpoint for portfolio chart data"""
    days = request.args.get('days', 30, type=int)
    return jsonify(data_provider.get_portfolio_chart_data(days))

@app.route('/api/positions')
def api_positions():
    """API endpoint for current positions"""
    return jsonify(data_provider.get_positions_data())

@app.route('/api/decisions')
def api_decisions():
    """API endpoint for recent trading decisions"""
    limit = request.args.get('limit', 10, type=int)
    days = request.args.get('days', type=int)
    action = request.args.get('action')
    symbol = request.args.get('symbol')
    
    return jsonify(data_provider.get_recent_decisions(limit, days, action, symbol))

@app.route('/api/bot/start', methods=['POST'])
def api_bot_start():
    """Start the trading bot"""
    global bot_instance, bot_thread, bot_status
    
    try:
        if bot_instance and bot_status['running']:
            return jsonify({'error': 'Bot is already running'}), 400
        
        # Get parameters from request
        data = request.get_json() or {}
        portfolio_name = data.get('portfolio_name', 'coinbase_all_eur')
        strategy_name = data.get('strategy_name', 'moderate')
        interval_minutes = data.get('interval_minutes', 15)
        
        # Create bot instance
        bot_instance = EnhancedMultiAssetBot(portfolio_name, strategy_name)
        
        # Start bot in separate thread
        def run_bot():
            global bot_status
            try:
                bot_status['running'] = True
                bot_status['error'] = None
                if bot_instance:
                    asyncio.run(bot_instance.start(interval_minutes))
            except Exception as e:
                logger.error(f"Bot error: {e}")
                bot_status['error'] = str(e)
                bot_status['running'] = False
        
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        
        bot_status['running'] = True
        bot_status['last_update'] = datetime.now().isoformat()
        
        return jsonify({'message': 'Bot started successfully'})
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        bot_status['error'] = str(e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/stop', methods=['POST'])
def api_bot_stop():
    """Stop the trading bot"""
    global bot_instance, bot_status
    
    try:
        if bot_instance:
            asyncio.run(bot_instance.stop())
            bot_instance = None
        
        bot_status['running'] = False
        bot_status['last_update'] = datetime.now().isoformat()
        
        return jsonify({'message': 'Bot stopped successfully'})
        
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/status')
def api_bot_status():
    """Get bot status"""
    global bot_status
    
    # Update status with current data
    if bot_instance:
        try:
            status = bot_instance.get_status()
            portfolio = status.get('portfolio', {})
            bot_status.update({
                'portfolio_value': portfolio.get('portfolio_value', 0),
                'total_pnl': portfolio.get('total_pnl', 0),
                'position_count': portfolio.get('position_count', 0),
                'last_update': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error updating bot status: {e}")
    
    return jsonify(bot_status)

@app.route('/api/bot/rebalance', methods=['POST'])
def api_bot_rebalance():
    """Trigger portfolio rebalancing"""
    global bot_instance, bot_status
    
    try:
        # Allow rebalancing even when bot is running - just use a fresh instance
        # This is safe because rebalancing uses its own analysis and doesn't interfere with the running bot
        
        # Import the CoinbaseSmartAllocationBot for rebalancing
        from coinbase_smart_allocation_bot import CoinbaseSmartAllocationBot
        
        # Create a rebalancing bot instance
        rebalance_bot = CoinbaseSmartAllocationBot('coinbase_all_eur', 'aggressive')
        
        # Run rebalancing in a separate thread
        def run_rebalancing():
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def rebalance():
                    await rebalance_bot.initialize()
                    success = await rebalance_bot.run_rebalancing_cycle()
                    return success
                
                success = loop.run_until_complete(rebalance())
                loop.close()
                
                if success:
                    logger.info("✅ Portfolio rebalancing completed successfully")
                    bot_status['last_rebalance'] = datetime.now().isoformat()
                else:
                    logger.error("❌ Portfolio rebalancing failed")
                    bot_status['error'] = 'Rebalancing failed'
                    
            except Exception as e:
                logger.error(f"Rebalancing error: {e}")
                bot_status['error'] = f'Rebalancing error: {str(e)}'
        
        rebalance_thread = threading.Thread(target=run_rebalancing, daemon=True)
        rebalance_thread.start()
        
        return jsonify({
            'message': 'Portfolio rebalancing initiated successfully',
            'status': 'running',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting rebalancing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs')
def api_logs():
    """Get recent log entries"""
    try:
        lines = request.args.get('lines', 100, type=int)
        level = request.args.get('level', 'all')  # all, error, warning, info, debug
        
        # Find the most recent log file
        log_files = []
        logs_dir = 'logs'
        
        if os.path.exists(logs_dir):
            for filename in os.listdir(logs_dir):
                if filename.endswith('.log'):
                    filepath = os.path.join(logs_dir, filename)
                    log_files.append((filepath, os.path.getmtime(filepath)))
        
        if not log_files:
            return jsonify({'logs': [], 'message': 'No log files found'})
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: x[1], reverse=True)
        most_recent_log = log_files[0][0]
        
        # Read the log file
        log_entries = []
        try:
            with open(most_recent_log, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
                # Get the last N lines
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                for line in recent_lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse log line (loguru format)
                    try:
                        # Extract timestamp, level, and message
                        if ' | ' in line:
                            parts = line.split(' | ')
                            if len(parts) >= 3:
                                timestamp = parts[0]
                                level_part = parts[1]
                                message = ' | '.join(parts[2:])
                                
                                # Extract level from level_part (e.g., "INFO     ")
                                log_level = level_part.split(':')[0].strip()
                                
                                # Filter by level if specified
                                if level != 'all' and log_level.lower() != level.lower():
                                    continue
                                
                                log_entries.append({
                                    'timestamp': timestamp,
                                    'level': log_level,
                                    'message': message
                                })
                    except Exception as e:
                        # If parsing fails, include the raw line
                        log_entries.append({
                            'timestamp': '',
                            'level': 'RAW',
                            'message': line
                        })
        
        except Exception as e:
            logger.error(f"Error reading log file {most_recent_log}: {e}")
            return jsonify({'logs': [], 'error': str(e)})
        
        return jsonify({
            'logs': log_entries,
            'log_file': most_recent_log,
            'total_lines': len(log_entries)
        })
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return jsonify({'logs': [], 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected to dashboard')
    emit('status', {'message': 'Connected to trading bot dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected from dashboard')

def broadcast_updates():
    """Broadcast real-time updates to all connected clients"""
    while True:
        try:
            # Get current data
            portfolio_data = data_provider.get_portfolio_summary()
            performance_data = data_provider.get_performance_metrics(7)  # Last 7 days
            risk_data = data_provider.get_risk_metrics()
            
            # Broadcast to all connected clients
            socketio.emit('portfolio_update', portfolio_data)
            socketio.emit('performance_update', performance_data)
            socketio.emit('risk_update', risk_data)
            
            # Update every 30 seconds
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"Error broadcasting updates: {e}")
            time.sleep(60)  # Wait longer on error

def start_dashboard(host='127.0.0.1', port=5000, debug=False):
    """Start the web dashboard"""
    logger.info(f"Starting trading bot dashboard at http://{host}:{port}")
    
    # Start background update thread
    update_thread = threading.Thread(target=broadcast_updates, daemon=True)
    update_thread.start()
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start Flask app
    socketio.run(app, host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_dashboard(debug=True) 