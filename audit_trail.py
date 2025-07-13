#!/usr/bin/env python3
"""
Audit Trail System for Trading Bot
Captures all LLM interactions and trading decisions for analysis
Enhanced with SQLite database storage alongside JSON files
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

class AuditTrail:
    def __init__(self, audit_folder: str = "audit_trails"):
        self.audit_folder = audit_folder
        os.makedirs(audit_folder, exist_ok=True)
        
        # Create subdirectories for different types of audits
        self.llm_interactions_dir = os.path.join(audit_folder, "llm_interactions")
        self.trading_decisions_dir = os.path.join(audit_folder, "trading_decisions")
        self.portfolio_snapshots_dir = os.path.join(audit_folder, "portfolio_snapshots")
        
        os.makedirs(self.llm_interactions_dir, exist_ok=True)
        os.makedirs(self.trading_decisions_dir, exist_ok=True)
        os.makedirs(self.portfolio_snapshots_dir, exist_ok=True)
        
        # Initialize SQLite database
        self.db_path = os.path.join(audit_folder, "audit_trail.db")
        self.init_database()
        
        logger.info(f"Audit trail system initialized in {audit_folder}")
    
    def init_database(self):
        """Initialize SQLite database for audit trail storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create LLM interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trading_phase TEXT NOT NULL,
                action TEXT,
                confidence REAL,
                reason TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                llm_response TEXT,
                parsed_decision TEXT,  -- JSON string
                market_data TEXT,      -- JSON string
                audit_id TEXT UNIQUE,
                filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create trading decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                decision_reason TEXT,
                current_price REAL,
                three_month_avg REAL,
                weekly_avg REAL,
                rsi REAL,
                macd REAL,
                volume_24h REAL,
                position_amount REAL,
                position_value REAL,
                position_pnl_pct REAL,
                position_pnl_eur REAL,
                execution_success BOOLEAN,
                execution_amount REAL,
                execution_price REAL,
                market_data TEXT,      -- JSON string
                position_data TEXT,    -- JSON string
                execution_result TEXT, -- JSON string
                audit_id TEXT UNIQUE,
                filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create portfolio snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                total_pnl REAL NOT NULL,
                available_cash REAL NOT NULL,
                position_count INTEGER NOT NULL,
                positions TEXT,        -- JSON string
                audit_id TEXT UNIQUE,
                filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_llm_timestamp ON llm_interactions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_llm_symbol ON llm_interactions(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_llm_action ON llm_interactions(action)')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON trading_decisions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON trading_decisions(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_action ON trading_decisions(action)')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_snapshots(timestamp)')
        
        conn.commit()
        conn.close()
    
    def log_llm_interaction(self, 
                          symbol: str, 
                          system_prompt: str, 
                          user_prompt: str, 
                          llm_response: str, 
                          parsed_decision: Dict[str, Any],
                          market_data: Dict[str, Any],
                          trading_phase: str) -> str:
        """
        Log a complete LLM interaction with all context
        Stores both in JSON file and SQLite database
        Returns the filename of the saved audit record
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        audit_record = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "trading_phase": trading_phase,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_response": llm_response,
            "parsed_decision": parsed_decision,
            "market_data_summary": self._summarize_market_data(market_data),
            "full_market_data": market_data,  # Keep full data for detailed analysis
            "audit_id": f"{symbol}_{timestamp_str}"
        }
        
        filename = f"llm_interaction_{symbol}_{timestamp_str}.json"
        filepath = os.path.join(self.llm_interactions_dir, filename)
        
        # Save to JSON file (existing functionality)
        with open(filepath, 'w') as f:
            json.dump(audit_record, f, indent=2, default=str)
        
        # Save to SQLite database (new functionality)
        self._store_llm_interaction_db(audit_record, filename)
        
        logger.info(f"📝 LLM interaction logged: {filename}")
        return filename
    
    def _store_llm_interaction_db(self, audit_record: Dict[str, Any], filename: str):
        """Store LLM interaction in SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            parsed_decision = audit_record.get('parsed_decision', {})
            
            cursor.execute('''
                INSERT INTO llm_interactions (
                    timestamp, symbol, trading_phase, action, confidence, reason,
                    system_prompt, user_prompt, llm_response, parsed_decision,
                    market_data, audit_id, filename
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_record['timestamp'],
                audit_record['symbol'],
                audit_record['trading_phase'],
                parsed_decision.get('action'),
                parsed_decision.get('confidence'),
                parsed_decision.get('reason'),
                audit_record['system_prompt'],
                audit_record['user_prompt'],
                audit_record['llm_response'],
                json.dumps(parsed_decision),
                json.dumps(audit_record['full_market_data']),
                audit_record['audit_id'],
                filename
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to store LLM interaction in database: {e}")
    
    def log_trading_decision(self, 
                           symbol: str, 
                           action: str, 
                           decision_reason: str,
                           market_data: Dict[str, Any],
                           position_data: Optional[Dict[str, Any]] = None,
                           execution_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a trading decision with execution details
        Stores both in JSON file and SQLite database
        Returns the filename of the saved audit record
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        audit_record = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "action": action,
            "decision_reason": decision_reason,
            "market_data_at_decision": self._summarize_market_data(market_data),
            "position_data": position_data,
            "execution_result": execution_result,
            "audit_id": f"{symbol}_{action}_{timestamp_str}"
        }
        
        filename = f"trading_decision_{symbol}_{action}_{timestamp_str}.json"
        filepath = os.path.join(self.trading_decisions_dir, filename)
        
        # Save to JSON file (existing functionality)
        with open(filepath, 'w') as f:
            json.dump(audit_record, f, indent=2, default=str)
        
        # Save to SQLite database (new functionality)
        self._store_trading_decision_db(audit_record, filename, market_data, position_data, execution_result)
        
        logger.info(f"📊 Trading decision logged: {filename}")
        return filename
    
    def _store_trading_decision_db(self, audit_record: Dict[str, Any], filename: str, 
                                 market_data: Dict[str, Any], position_data: Optional[Dict[str, Any]], 
                                 execution_result: Optional[Dict[str, Any]]):
        """Store trading decision in SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract key market data fields
            current_price = market_data.get('current_price', 0)
            three_month_avg = market_data.get('three_month_average', 0)
            weekly_avg = market_data.get('weekly_average', 0)
            rsi = market_data.get('rsi')
            macd = market_data.get('macd')
            volume_24h = market_data.get('volume_24h')
            
            # Extract position data
            position_amount = 0
            position_value = 0
            position_pnl_pct = 0
            position_pnl_eur = 0
            
            if position_data:
                position_amount = position_data.get('amount', 0)
                position_value = position_data.get('eur_value', 0)
                position_pnl_pct = position_data.get('profit_loss_pct', 0)
                position_pnl_eur = position_data.get('profit_loss_eur', 0)
            
            # Extract execution data
            execution_success = False
            execution_amount = 0
            execution_price = 0
            
            if execution_result:
                execution_success = execution_result.get('success', False)
                execution_amount = execution_result.get('eur_amount', 0) or execution_result.get('crypto_amount', 0)
                execution_price = execution_result.get('buy_price', 0) or current_price
            
            cursor.execute('''
                INSERT INTO trading_decisions (
                    timestamp, symbol, action, decision_reason, current_price,
                    three_month_avg, weekly_avg, rsi, macd, volume_24h,
                    position_amount, position_value, position_pnl_pct, position_pnl_eur,
                    execution_success, execution_amount, execution_price,
                    market_data, position_data, execution_result, audit_id, filename
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_record['timestamp'],
                audit_record['symbol'],
                audit_record['action'],
                audit_record['decision_reason'],
                current_price,
                three_month_avg,
                weekly_avg,
                rsi,
                macd,
                volume_24h,
                position_amount,
                position_value,
                position_pnl_pct,
                position_pnl_eur,
                execution_success,
                execution_amount,
                execution_price,
                json.dumps(market_data),
                json.dumps(position_data),
                json.dumps(execution_result),
                audit_record['audit_id'],
                filename
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to store trading decision in database: {e}")
    
    def log_portfolio_snapshot(self, 
                             portfolio_value: float,
                             positions: Dict[str, Any],
                             total_pnl: float,
                             available_cash: float) -> str:
        """
        Log a portfolio snapshot for tracking changes over time
        Stores both in JSON file and SQLite database
        Returns the filename of the saved audit record
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        audit_record = {
            "timestamp": timestamp.isoformat(),
            "portfolio_value": portfolio_value,
            "total_pnl": total_pnl,
            "available_cash": available_cash,
            "position_count": len(positions),
            "positions": positions,
            "audit_id": f"portfolio_{timestamp_str}"
        }
        
        filename = f"portfolio_snapshot_{timestamp_str}.json"
        filepath = os.path.join(self.portfolio_snapshots_dir, filename)
        
        # Save to JSON file (existing functionality)
        with open(filepath, 'w') as f:
            json.dump(audit_record, f, indent=2, default=str)
        
        # Save to SQLite database (new functionality)
        self._store_portfolio_snapshot_db(audit_record, filename)
        
        logger.info(f"📈 Portfolio snapshot logged: {filename}")
        return filename
    
    def _store_portfolio_snapshot_db(self, audit_record: Dict[str, Any], filename: str):
        """Store portfolio snapshot in SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_snapshots (
                    timestamp, portfolio_value, total_pnl, available_cash,
                    position_count, positions, audit_id, filename
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_record['timestamp'],
                audit_record['portfolio_value'],
                audit_record['total_pnl'],
                audit_record['available_cash'],
                audit_record['position_count'],
                json.dumps(audit_record['positions']),
                audit_record['audit_id'],
                filename
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to store portfolio snapshot in database: {e}")
    
    def _summarize_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of market data for audit records"""
        summary = {
            "symbol": market_data.get('symbol'),
            "current_price": market_data.get('current_price'),
            "three_month_average": market_data.get('three_month_average'),
            "weekly_average": market_data.get('weekly_average'),
            "rsi": market_data.get('rsi'),
            "macd": market_data.get('macd'),
            "volume_24h": market_data.get('volume_24h'),
            "ma_20": market_data.get('ma_20'),
            "ma_50": market_data.get('ma_50'),
        }
        
        # Add position info if available
        current_position = market_data.get('current_position')
        if current_position:
            summary["position"] = {
                "has_position": current_position.get('has_position'),
                "amount": current_position.get('amount'),
                "eur_value": current_position.get('eur_value'),
                "buy_price": current_position.get('buy_price'),
                "profit_loss_pct": current_position.get('profit_loss_pct'),
            }
        
        # Add profit/loss info if available
        if market_data.get('buy_price'):
            summary["profit_loss"] = {
                "buy_price": market_data.get('buy_price'),
                "profit_loss_pct": market_data.get('profit_loss_pct'),
                "profit_loss_eur": market_data.get('profit_loss_eur'),
            }
        
        return summary
    
    # Database query methods
    def get_recent_decisions_db(self, symbol: Optional[str] = None, action: Optional[str] = None, 
                               days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trading decisions from database with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query with optional filters
            query = '''
                SELECT * FROM trading_decisions 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days)
            
            params = []
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            if action:
                query += ' AND action = ?'
                params.append(action)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                decision = dict(zip(columns, row))
                # Parse JSON fields
                if decision['market_data']:
                    decision['market_data'] = json.loads(decision['market_data'])
                if decision['position_data']:
                    decision['position_data'] = json.loads(decision['position_data'])
                if decision['execution_result']:
                    decision['execution_result'] = json.loads(decision['execution_result'])
                results.append(decision)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error querying decisions from database: {e}")
            return []
    
    def get_recent_llm_interactions_db(self, symbol: Optional[str] = None, action: Optional[str] = None,
                                      days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent LLM interactions from database with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query with optional filters
            query = '''
                SELECT * FROM llm_interactions 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days)
            
            params = []
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            if action:
                query += ' AND action = ?'
                params.append(action)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                interaction = dict(zip(columns, row))
                # Parse JSON fields
                if interaction['parsed_decision']:
                    interaction['parsed_decision'] = json.loads(interaction['parsed_decision'])
                if interaction['market_data']:
                    interaction['market_data'] = json.loads(interaction['market_data'])
                results.append(interaction)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error querying LLM interactions from database: {e}")
            return []
    
    def get_portfolio_history_db(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get portfolio snapshots from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM portfolio_snapshots 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days))
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                snapshot = dict(zip(columns, row))
                # Parse JSON fields
                if snapshot['positions']:
                    snapshot['positions'] = json.loads(snapshot['positions'])
                results.append(snapshot)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error querying portfolio history from database: {e}")
            return []
    
    def get_trading_statistics_db(self, symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get trading statistics from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build base query
            base_query = '''
                SELECT action, COUNT(*) as count, AVG(execution_amount) as avg_amount,
                       SUM(CASE WHEN execution_success = 1 THEN 1 ELSE 0 END) as successful_trades
                FROM trading_decisions 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days)
            
            if symbol:
                base_query += ' AND symbol = ?'
                cursor.execute(base_query + ' GROUP BY action', (symbol,))
            else:
                cursor.execute(base_query + ' GROUP BY action')
            
            results = cursor.fetchall()
            
            stats = {
                'total_decisions': 0,
                'buy_decisions': 0,
                'sell_decisions': 0,
                'hold_decisions': 0,
                'successful_trades': 0,
                'avg_trade_amount': 0
            }
            
            for action, count, avg_amount, successful in results:
                stats['total_decisions'] += count
                if action == 'BUY':
                    stats['buy_decisions'] = count
                elif action == 'SELL':
                    stats['sell_decisions'] = count
                elif action == 'HOLD':
                    stats['hold_decisions'] = count
                
                stats['successful_trades'] += successful
                if avg_amount:
                    stats['avg_trade_amount'] += avg_amount
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error querying trading statistics from database: {e}")
            return {}
    
    # Existing JSON-based methods (maintained for backward compatibility)
    def get_recent_decisions(self, symbol: Optional[str] = None, action: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading decisions with optional filtering (JSON files)"""
        decisions = []
        
        # Check if directory exists
        if not os.path.exists(self.trading_decisions_dir):
            logger.info(f"Trading decisions directory does not exist: {self.trading_decisions_dir}")
            return decisions
        
        for filename in os.listdir(self.trading_decisions_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.trading_decisions_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    decision = json.load(f)
                
                # Apply filters
                if symbol and decision.get('symbol') != symbol:
                    continue
                if action and decision.get('action') != action:
                    continue
                
                decisions.append(decision)
            except Exception as e:
                logger.warning(f"Could not load decision file {filename}: {e}")
        
        # Sort by timestamp and limit results
        decisions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return decisions[:limit]
    
    def get_llm_interaction_by_audit_id(self, audit_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific LLM interaction by audit ID"""
        for filename in os.listdir(self.llm_interactions_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.llm_interactions_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    interaction = json.load(f)
                
                if interaction.get('audit_id') == audit_id:
                    return interaction
            except Exception as e:
                logger.warning(f"Could not load interaction file {filename}: {e}")
        
        return None
    
    def generate_audit_report(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive audit report for a date range"""
        report = {
            "report_generated": datetime.now().isoformat(),
            "date_range": {"start": start_date, "end": end_date},
            "summary": {},
            "decisions": [],
            "llm_interactions": []
        }
        
        # Collect all decisions in date range
        all_decisions = []
        for filename in os.listdir(self.trading_decisions_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.trading_decisions_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    decision = json.load(f)
                
                decision_date = decision.get('timestamp', '')[:10]  # YYYY-MM-DD
                if start_date and decision_date < start_date:
                    continue
                if end_date and decision_date > end_date:
                    continue
                
                all_decisions.append(decision)
            except Exception as e:
                logger.warning(f"Could not load decision file {filename}: {e}")
        
        # Generate summary statistics
        actions = [d.get('action') for d in all_decisions]
        symbols = [d.get('symbol') for d in all_decisions]
        
        report["summary"] = {
            "total_decisions": len(all_decisions),
            "buy_decisions": actions.count('BUY'),
            "sell_decisions": actions.count('SELL'),
            "hold_decisions": actions.count('HOLD'),
            "unique_symbols": len(set(symbols)),
            "most_traded_symbols": self._get_most_frequent(symbols, 5)
        }
        
        report["decisions"] = all_decisions
        
        return report
    
    def _get_most_frequent(self, items: List[str], limit: int) -> List[Dict[str, Any]]:
        """Get the most frequent items in a list"""
        from collections import Counter
        counter = Counter(items)
        return [{"item": item, "count": count} for item, count in counter.most_common(limit)]
    
    def cleanup_old_audits(self, days_to_keep: int = 30):
        """Clean up audit files older than specified days"""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        # Clean up JSON files
        for directory in [self.llm_interactions_dir, self.trading_decisions_dir, self.portfolio_snapshots_dir]:
            for filename in os.listdir(directory):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(directory, filename)
                file_time = os.path.getmtime(filepath)
                
                if file_time < cutoff_date:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old audit file: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not remove old audit file {filename}: {e}")
        
        # Clean up database records
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
            
            cursor.execute('DELETE FROM llm_interactions WHERE timestamp < ?', (cutoff_iso,))
            cursor.execute('DELETE FROM trading_decisions WHERE timestamp < ?', (cutoff_iso,))
            cursor.execute('DELETE FROM portfolio_snapshots WHERE timestamp < ?', (cutoff_iso,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up database records older than {days_to_keep} days")
            
        except Exception as e:
            logger.warning(f"Could not clean up database records: {e}")

# Global audit trail instance
audit_trail = AuditTrail() 