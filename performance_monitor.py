#!/usr/bin/env python3
"""
Performance Monitoring System for Trading Bot
Tracks trading metrics, portfolio performance, and generates analytics reports
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
import sqlite3
from pathlib import Path

# Ensure performance data directory exists
PERFORMANCE_DATA_DIR = "performance_data"
os.makedirs(PERFORMANCE_DATA_DIR, exist_ok=True)

@dataclass
class TradeRecord:
    """Individual trade record"""
    timestamp: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    price: float
    quantity: float
    value: float
    profit_loss: float
    profit_loss_pct: float
    reason: str
    confidence: float
    rsi: Optional[float] = None
    macd: Optional[float] = None
    volume: Optional[float] = None

@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at a point in time"""
    timestamp: str
    total_value: float
    cash_balance: float
    positions_value: float
    total_pnl: float
    daily_pnl: float
    position_count: int
    positions: Dict[str, Dict[str, Any]]
    
@dataclass
class PerformanceMetrics:
    """Calculated performance metrics"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    best_trade: float
    worst_trade: float
    avg_trade_duration: float
    volatility: float

class PerformanceMonitor:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(PERFORMANCE_DATA_DIR, "performance.db")
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for performance tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                value REAL NOT NULL,
                profit_loss REAL DEFAULT 0,
                profit_loss_pct REAL DEFAULT 0,
                reason TEXT,
                confidence REAL,
                rsi REAL,
                macd REAL,
                volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create portfolio snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                positions_value REAL NOT NULL,
                total_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                position_count INTEGER NOT NULL,
                positions TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_return REAL,
                annualized_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                avg_win REAL,
                avg_loss REAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                best_trade REAL,
                worst_trade REAL,
                avg_trade_duration REAL,
                volatility REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def record_trade(self, trade: TradeRecord):
        """Record a trade in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    timestamp, symbol, action, price, quantity, value,
                    profit_loss, profit_loss_pct, reason, confidence, rsi, macd, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.timestamp, trade.symbol, trade.action, trade.price, trade.quantity,
                trade.value, trade.profit_loss, trade.profit_loss_pct, trade.reason,
                trade.confidence, trade.rsi, trade.macd, trade.volume
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Recorded trade: {trade.action} {trade.symbol} at {trade.price}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def record_portfolio_snapshot(self, snapshot: PortfolioSnapshot):
        """Record a portfolio snapshot in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_snapshots (
                    timestamp, total_value, cash_balance, positions_value,
                    total_pnl, daily_pnl, position_count, positions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp, snapshot.total_value, snapshot.cash_balance,
                snapshot.positions_value, snapshot.total_pnl, snapshot.daily_pnl,
                snapshot.position_count, json.dumps(snapshot.positions)
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Recorded portfolio snapshot: €{snapshot.total_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording portfolio snapshot: {e}")
    
    def calculate_performance_metrics(self, days: int = 30) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get portfolio snapshots for the period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            snapshots_df = pd.read_sql_query('''
                SELECT * FROM portfolio_snapshots 
                WHERE timestamp >= ? 
                ORDER BY timestamp
            ''', conn, params=(start_date.isoformat(),))
            
            # Get trades for the period
            trades_df = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE timestamp >= ? AND action IN ('BUY', 'SELL')
                ORDER BY timestamp
            ''', conn, params=(start_date.isoformat(),))
            
            conn.close()
            
            if snapshots_df.empty or trades_df.empty:
                logger.warning("Insufficient data for performance calculation")
                return self._empty_metrics()
            
            # Calculate returns
            snapshots_df['timestamp'] = pd.to_datetime(snapshots_df['timestamp'])
            snapshots_df = snapshots_df.sort_values('timestamp')
            
            initial_value = snapshots_df['total_value'].iloc[0]
            final_value = snapshots_df['total_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate daily returns
            snapshots_df['daily_return'] = snapshots_df['total_value'].pct_change()
            daily_returns = snapshots_df['daily_return'].dropna()
            
            # Annualized return
            days_elapsed = (snapshots_df['timestamp'].iloc[-1] - snapshots_df['timestamp'].iloc[0]).days
            annualized_return = (1 + total_return) ** (365 / max(days_elapsed, 1)) - 1
            
            # Sharpe ratio (assuming 0% risk-free rate)
            volatility = daily_returns.std() * np.sqrt(365)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            # Trade statistics
            profitable_trades = trades_df[trades_df['profit_loss'] > 0]
            losing_trades = trades_df[trades_df['profit_loss'] < 0]
            
            win_rate = len(profitable_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
            avg_win = profitable_trades['profit_loss'].mean() if len(profitable_trades) > 0 else 0
            avg_loss = abs(losing_trades['profit_loss'].mean()) if len(losing_trades) > 0 else 0
            
            profit_factor = (profitable_trades['profit_loss'].sum() / abs(losing_trades['profit_loss'].sum())) if len(losing_trades) > 0 and losing_trades['profit_loss'].sum() != 0 else float('inf')
            
            best_trade = trades_df['profit_loss'].max() if len(trades_df) > 0 else 0
            worst_trade = trades_df['profit_loss'].min() if len(trades_df) > 0 else 0
            
            # Average trade duration (placeholder - would need entry/exit matching)
            avg_trade_duration = 24.0  # hours, placeholder
            
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                total_trades=len(trades_df),
                winning_trades=len(profitable_trades),
                losing_trades=len(losing_trades),
                best_trade=best_trade,
                worst_trade=worst_trade,
                avg_trade_duration=avg_trade_duration,
                volatility=volatility
            )
            
            # Store metrics
            self._store_performance_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics"""
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, total_trades=0,
            winning_trades=0, losing_trades=0, best_trade=0.0,
            worst_trade=0.0, avg_trade_duration=0.0, volatility=0.0
        )
    
    def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (
                    timestamp, total_return, annualized_return, sharpe_ratio,
                    max_drawdown, win_rate, profit_factor, avg_win, avg_loss,
                    total_trades, winning_trades, losing_trades, best_trade,
                    worst_trade, avg_trade_duration, volatility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metrics.total_return, metrics.annualized_return, metrics.sharpe_ratio,
                metrics.max_drawdown, metrics.win_rate, metrics.profit_factor,
                metrics.avg_win, metrics.avg_loss, metrics.total_trades,
                metrics.winning_trades, metrics.losing_trades, metrics.best_trade,
                metrics.worst_trade, metrics.avg_trade_duration, metrics.volatility
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
    
    def generate_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            metrics = self.calculate_performance_metrics(days)
            
            # Get recent trades
            conn = sqlite3.connect(self.db_path)
            recent_trades = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''', conn, params=((datetime.now() - timedelta(days=7)).isoformat(),))
            
            # Get portfolio evolution
            portfolio_evolution = pd.read_sql_query('''
                SELECT timestamp, total_value, total_pnl 
                FROM portfolio_snapshots 
                WHERE timestamp >= ? 
                ORDER BY timestamp
            ''', conn, params=((datetime.now() - timedelta(days=days)).isoformat(),))
            
            conn.close()
            
            report = {
                'report_date': datetime.now().isoformat(),
                'period_days': days,
                'performance_metrics': asdict(metrics),
                'recent_trades': recent_trades.to_dict('records') if not recent_trades.empty else [],
                'portfolio_evolution': portfolio_evolution.to_dict('records') if not portfolio_evolution.empty else [],
                'summary': {
                    'total_return_pct': metrics.total_return * 100,
                    'annualized_return_pct': metrics.annualized_return * 100,
                    'win_rate_pct': metrics.win_rate * 100,
                    'max_drawdown_pct': metrics.max_drawdown * 100,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'profit_factor': metrics.profit_factor,
                    'total_trades': metrics.total_trades,
                    'avg_win_loss_ratio': metrics.avg_win / metrics.avg_loss if metrics.avg_loss > 0 else 0
                }
            }
            
            # Save report to file
            report_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = os.path.join(PERFORMANCE_DATA_DIR, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report generated: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    def get_trading_statistics(self, symbol: str = None, days: int = 30) -> Dict[str, Any]:
        """Get trading statistics for a specific symbol or all symbols"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT symbol, action, COUNT(*) as count, 
                       AVG(profit_loss) as avg_pnl,
                       SUM(profit_loss) as total_pnl,
                       AVG(confidence) as avg_confidence
                FROM trades 
                WHERE timestamp >= ?
            '''
            params = [(datetime.now() - timedelta(days=days)).isoformat()]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " GROUP BY symbol, action ORDER BY symbol, action"
            
            stats_df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return stats_df.to_dict('records') if not stats_df.empty else []
            
        except Exception as e:
            logger.error(f"Error getting trading statistics: {e}")
            return []
    
    def get_portfolio_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get portfolio value history"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            history_df = pd.read_sql_query('''
                SELECT timestamp, total_value, cash_balance, positions_value, total_pnl, daily_pnl
                FROM portfolio_snapshots 
                WHERE timestamp >= ? 
                ORDER BY timestamp
            ''', conn, params=((datetime.now() - timedelta(days=days)).isoformat(),))
            
            conn.close()
            
            return history_df.to_dict('records') if not history_df.empty else []
            
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return []
    
    def print_performance_summary(self, days: int = 30):
        """Print a formatted performance summary"""
        metrics = self.calculate_performance_metrics(days)
        
        print("\n" + "="*60)
        print(f"PERFORMANCE SUMMARY - Last {days} Days")
        print("="*60)
        print(f"Total Return:        {metrics.total_return*100:+.2f}%")
        print(f"Annualized Return:   {metrics.annualized_return*100:+.2f}%")
        print(f"Sharpe Ratio:        {metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown:        {metrics.max_drawdown*100:.2f}%")
        print(f"Volatility:          {metrics.volatility*100:.2f}%")
        print("-"*60)
        print(f"Total Trades:        {metrics.total_trades}")
        print(f"Win Rate:            {metrics.win_rate*100:.1f}%")
        print(f"Winning Trades:      {metrics.winning_trades}")
        print(f"Losing Trades:       {metrics.losing_trades}")
        print(f"Profit Factor:       {metrics.profit_factor:.2f}")
        print("-"*60)
        print(f"Average Win:         €{metrics.avg_win:.2f}")
        print(f"Average Loss:        €{metrics.avg_loss:.2f}")
        print(f"Best Trade:          €{metrics.best_trade:.2f}")
        print(f"Worst Trade:         €{metrics.worst_trade:.2f}")
        print("="*60)
    
    def get_recent_trades(self, limit: int = 20) -> List[TradeRecord]:
        """Get recent trades as TradeRecord objects"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, symbol, action, price, quantity, value,
                       profit_loss, profit_loss_pct, reason, confidence, rsi, macd, volume
                FROM trades 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            trades = []
            for row in cursor.fetchall():
                trade = TradeRecord(
                    timestamp=row[0],
                    symbol=row[1],
                    action=row[2],
                    price=row[3],
                    quantity=row[4],
                    value=row[5],
                    profit_loss=row[6],
                    profit_loss_pct=row[7],
                    reason=row[8],
                    confidence=row[9],
                    rsi=row[10],
                    macd=row[11],
                    volume=row[12]
                )
                trades.append(trade)
            
            conn.close()
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def get_portfolio_snapshots(self, days: int = 30) -> List[PortfolioSnapshot]:
        """Get portfolio snapshots for the last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
                SELECT timestamp, total_value, cash_balance, positions_value,
                       total_pnl, daily_pnl, position_count, positions
                FROM portfolio_snapshots 
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            ''', (start_date,))
            
            snapshots = []
            for row in cursor.fetchall():
                positions = json.loads(row[7]) if row[7] else {}
                snapshot = PortfolioSnapshot(
                    timestamp=row[0],
                    total_value=row[1],
                    cash_balance=row[2],
                    positions_value=row[3],
                    total_pnl=row[4],
                    daily_pnl=row[5],
                    position_count=row[6],
                    positions=positions
                )
                snapshots.append(snapshot)
            
            conn.close()
            return snapshots
            
        except Exception as e:
            logger.error(f"Error getting portfolio snapshots: {e}")
            return []

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def record_trade_from_audit(audit_data: Dict[str, Any]):
    """Helper function to record trade from audit trail data"""
    try:
        trade = TradeRecord(
            timestamp=audit_data.get('timestamp', datetime.now().isoformat()),
            symbol=audit_data.get('symbol', ''),
            action=audit_data.get('action', 'HOLD'),
            price=audit_data.get('market_data', {}).get('current_price', 0),
            quantity=audit_data.get('execution_result', {}).get('crypto_amount', 0),
            value=audit_data.get('execution_result', {}).get('eur_amount', 0),
            profit_loss=audit_data.get('position_data', {}).get('profit_loss_eur', 0),
            profit_loss_pct=audit_data.get('position_data', {}).get('profit_loss_pct', 0),
            reason=audit_data.get('decision_reason', ''),
            confidence=audit_data.get('market_data', {}).get('confidence', 50),
            rsi=audit_data.get('market_data', {}).get('rsi'),
            macd=audit_data.get('market_data', {}).get('macd'),
            volume=audit_data.get('market_data', {}).get('volume_24h')
        )
        
        performance_monitor.record_trade(trade)
        
    except Exception as e:
        logger.error(f"Error recording trade from audit data: {e}")

def record_portfolio_from_audit(audit_data: Dict[str, Any]):
    """Helper function to record portfolio snapshot from audit trail data"""
    try:
        snapshot = PortfolioSnapshot(
            timestamp=audit_data.get('timestamp', datetime.now().isoformat()),
            total_value=audit_data.get('portfolio_value', 0),
            cash_balance=audit_data.get('available_cash', 0),
            positions_value=sum(pos.get('eur_value', 0) for pos in audit_data.get('positions', {}).values()),
            total_pnl=audit_data.get('total_pnl', 0),
            daily_pnl=0,  # Would need to calculate
            position_count=len(audit_data.get('positions', {})),
            positions=audit_data.get('positions', {})
        )
        
        performance_monitor.record_portfolio_snapshot(snapshot)
        
    except Exception as e:
        logger.error(f"Error recording portfolio snapshot from audit data: {e}") 