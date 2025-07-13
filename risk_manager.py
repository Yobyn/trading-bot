#!/usr/bin/env python3
"""
Advanced Risk Management System for Trading Bot
Provides correlation analysis, position sizing optimization, and dynamic risk controls
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
from scipy.stats import pearsonr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Ensure risk data directory exists
RISK_DATA_DIR = "risk_data"
os.makedirs(RISK_DATA_DIR, exist_ok=True)

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio analysis"""
    portfolio_var: float  # Value at Risk
    portfolio_cvar: float  # Conditional Value at Risk
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    overall_risk_score: float

@dataclass
class PositionRisk:
    """Risk metrics for individual positions"""
    symbol: str
    position_size: float
    value_at_risk: float
    beta: float
    correlation_with_portfolio: float
    liquidity_score: float
    concentration_pct: float
    risk_contribution: float
    recommended_size: float

@dataclass
class RiskAlert:
    """Risk alert for monitoring"""
    timestamp: str
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    symbol: Optional[str]
    message: str
    current_value: float
    threshold: float
    recommendation: str

class RiskManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(RISK_DATA_DIR, "risk_data.db")
        self.init_database()
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_portfolio_var': 0.05,  # 5% daily VaR
            'max_position_concentration': 0.25,  # 25% max position size
            'max_correlation': 0.8,  # 80% max correlation
            'min_liquidity_score': 0.3,  # 30% min liquidity
            'max_drawdown': 0.15,  # 15% max drawdown
            'min_sharpe_ratio': 0.5,  # 0.5 min Sharpe ratio
            'max_beta': 2.0,  # 2.0 max beta
            'max_volatility': 0.4  # 40% max annualized volatility
        }
        
        # Risk weights for overall score
        self.risk_weights = {
            'var': 0.25,
            'correlation': 0.20,
            'concentration': 0.20,
            'liquidity': 0.15,
            'volatility': 0.10,
            'drawdown': 0.10
        }
        
    def init_database(self):
        """Initialize SQLite database for risk tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create risk metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_var REAL,
                portfolio_cvar REAL,
                max_drawdown REAL,
                volatility REAL,
                sharpe_ratio REAL,
                beta REAL,
                correlation_risk REAL,
                concentration_risk REAL,
                liquidity_risk REAL,
                overall_risk_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create position risk table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_risk (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                position_size REAL,
                value_at_risk REAL,
                beta REAL,
                correlation_with_portfolio REAL,
                liquidity_score REAL,
                concentration_pct REAL,
                risk_contribution REAL,
                recommended_size REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create risk alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                symbol TEXT,
                message TEXT,
                current_value REAL,
                threshold REAL,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_portfolio_risk(self, positions: Dict[str, Any], 
                               price_history: Dict[str, List[float]]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if not positions or not price_history:
                return self._empty_risk_metrics()
            
            # Calculate returns for each asset
            returns_data = {}
            for symbol in positions.keys():
                if symbol in price_history and len(price_history[symbol]) > 1:
                    prices = np.array(price_history[symbol])
                    returns = np.diff(prices) / prices[:-1]
                    returns_data[symbol] = returns
            
            if not returns_data:
                return self._empty_risk_metrics()
            
            # Create returns DataFrame
            min_length = min(len(returns) for returns in returns_data.values())
            returns_df = pd.DataFrame({
                symbol: returns[-min_length:] 
                for symbol, returns in returns_data.items()
            })
            
            # Calculate portfolio weights
            total_value = sum(pos.get('eur_value', 0) for pos in positions.values())
            weights = np.array([
                positions[symbol].get('eur_value', 0) / total_value 
                for symbol in returns_df.columns
            ])
            
            # Portfolio returns
            portfolio_returns = np.dot(returns_df.values, weights)
            
            # Calculate risk metrics
            portfolio_var = self._calculate_var(portfolio_returns)
            portfolio_cvar = self._calculate_cvar(portfolio_returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
            
            # Beta calculation (vs market proxy - could be BTC)
            beta = self._calculate_beta(portfolio_returns, returns_df)
            
            # Correlation risk
            correlation_matrix = returns_df.corr()
            correlation_risk = self._calculate_correlation_risk(correlation_matrix, weights)
            
            # Concentration risk
            concentration_risk = self._calculate_concentration_risk(weights)
            
            # Liquidity risk (placeholder - would need volume data)
            liquidity_risk = 0.5  # Medium risk assumption
            
            # Overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                portfolio_var, correlation_risk, concentration_risk, 
                liquidity_risk, volatility, max_drawdown
            )
            
            metrics = RiskMetrics(
                portfolio_var=portfolio_var,
                portfolio_cvar=portfolio_cvar,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                overall_risk_score=overall_risk_score
            )
            
            # Store metrics
            self._store_risk_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return self._empty_risk_metrics()
    
    def calculate_position_risk(self, symbol: str, position: Dict[str, Any], 
                              portfolio_positions: Dict[str, Any],
                              price_history: Dict[str, List[float]]) -> PositionRisk:
        """Calculate risk metrics for individual position"""
        try:
            position_size = position.get('eur_value', 0)
            total_portfolio_value = sum(pos.get('eur_value', 0) for pos in portfolio_positions.values())
            
            # Value at Risk for position
            if symbol in price_history and len(price_history[symbol]) > 1:
                prices = np.array(price_history[symbol])
                returns = np.diff(prices) / prices[:-1]
                value_at_risk = self._calculate_var(returns) * position_size
            else:
                value_at_risk = 0
            
            # Beta calculation
            beta = self._calculate_position_beta(symbol, price_history)
            
            # Correlation with portfolio
            correlation_with_portfolio = self._calculate_position_correlation(
                symbol, portfolio_positions, price_history
            )
            
            # Liquidity score (placeholder)
            liquidity_score = 0.7  # Assume good liquidity
            
            # Concentration percentage
            concentration_pct = position_size / total_portfolio_value if total_portfolio_value > 0 else 0
            
            # Risk contribution
            risk_contribution = concentration_pct * abs(value_at_risk)
            
            # Recommended size based on risk
            recommended_size = self._calculate_optimal_position_size(
                symbol, portfolio_positions, price_history
            )
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_size=position_size,
                value_at_risk=value_at_risk,
                beta=beta,
                correlation_with_portfolio=correlation_with_portfolio,
                liquidity_score=liquidity_score,
                concentration_pct=concentration_pct,
                risk_contribution=risk_contribution,
                recommended_size=recommended_size
            )
            
            # Store position risk
            self._store_position_risk(position_risk)
            
            return position_risk
            
        except Exception as e:
            logger.error(f"Error calculating position risk for {symbol}: {e}")
            return PositionRisk(
                symbol=symbol, position_size=0, value_at_risk=0, beta=1.0,
                correlation_with_portfolio=0, liquidity_score=0.5,
                concentration_pct=0, risk_contribution=0, recommended_size=0
            )
    
    def check_risk_alerts(self, portfolio_risk: RiskMetrics, 
                         position_risks: List[PositionRisk]) -> List[RiskAlert]:
        """Check for risk threshold violations and generate alerts"""
        alerts = []
        timestamp = datetime.now().isoformat()
        
        # Portfolio-level alerts
        if portfolio_risk.portfolio_var > self.risk_thresholds['max_portfolio_var']:
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='PORTFOLIO_VAR',
                severity='HIGH',
                symbol=None,
                message=f"Portfolio VaR ({portfolio_risk.portfolio_var:.2%}) exceeds threshold",
                current_value=portfolio_risk.portfolio_var,
                threshold=self.risk_thresholds['max_portfolio_var'],
                recommendation="Reduce position sizes or diversify holdings"
            ))
        
        if portfolio_risk.max_drawdown > self.risk_thresholds['max_drawdown']:
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='MAX_DRAWDOWN',
                severity='HIGH',
                symbol=None,
                message=f"Max drawdown ({portfolio_risk.max_drawdown:.2%}) exceeds threshold",
                current_value=portfolio_risk.max_drawdown,
                threshold=self.risk_thresholds['max_drawdown'],
                recommendation="Review stop-loss settings and position sizing"
            ))
        
        if portfolio_risk.volatility > self.risk_thresholds['max_volatility']:
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='VOLATILITY',
                severity='MEDIUM',
                symbol=None,
                message=f"Portfolio volatility ({portfolio_risk.volatility:.2%}) exceeds threshold",
                current_value=portfolio_risk.volatility,
                threshold=self.risk_thresholds['max_volatility'],
                recommendation="Consider more stable assets or reduce leverage"
            ))
        
        if portfolio_risk.sharpe_ratio < self.risk_thresholds['min_sharpe_ratio']:
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='SHARPE_RATIO',
                severity='MEDIUM',
                symbol=None,
                message=f"Sharpe ratio ({portfolio_risk.sharpe_ratio:.2f}) below threshold",
                current_value=portfolio_risk.sharpe_ratio,
                threshold=self.risk_thresholds['min_sharpe_ratio'],
                recommendation="Improve risk-adjusted returns through better asset selection"
            ))
        
        # Position-level alerts
        for pos_risk in position_risks:
            if pos_risk.concentration_pct > self.risk_thresholds['max_position_concentration']:
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    alert_type='CONCENTRATION',
                    severity='HIGH',
                    symbol=pos_risk.symbol,
                    message=f"{pos_risk.symbol} concentration ({pos_risk.concentration_pct:.2%}) exceeds threshold",
                    current_value=pos_risk.concentration_pct,
                    threshold=self.risk_thresholds['max_position_concentration'],
                    recommendation=f"Reduce {pos_risk.symbol} position size"
                ))
            
            if abs(pos_risk.correlation_with_portfolio) > self.risk_thresholds['max_correlation']:
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    alert_type='CORRELATION',
                    severity='MEDIUM',
                    symbol=pos_risk.symbol,
                    message=f"{pos_risk.symbol} correlation ({pos_risk.correlation_with_portfolio:.2f}) exceeds threshold",
                    current_value=abs(pos_risk.correlation_with_portfolio),
                    threshold=self.risk_thresholds['max_correlation'],
                    recommendation=f"Diversify away from {pos_risk.symbol} or similar assets"
                ))
            
            if pos_risk.liquidity_score < self.risk_thresholds['min_liquidity_score']:
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    alert_type='LIQUIDITY',
                    severity='MEDIUM',
                    symbol=pos_risk.symbol,
                    message=f"{pos_risk.symbol} liquidity score ({pos_risk.liquidity_score:.2f}) below threshold",
                    current_value=pos_risk.liquidity_score,
                    threshold=self.risk_thresholds['min_liquidity_score'],
                    recommendation=f"Monitor {pos_risk.symbol} for exit opportunities"
                ))
        
        # Store alerts
        for alert in alerts:
            self._store_risk_alert(alert)
        
        return alerts
    
    def optimize_portfolio_allocation(self, available_assets: List[str], 
                                    price_history: Dict[str, List[float]],
                                    target_return: float = 0.1) -> Dict[str, float]:
        """Optimize portfolio allocation using Modern Portfolio Theory"""
        try:
            # Calculate returns for each asset
            returns_data = {}
            for symbol in available_assets:
                if symbol in price_history and len(price_history[symbol]) > 1:
                    prices = np.array(price_history[symbol])
                    returns = np.diff(prices) / prices[:-1]
                    returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                # Not enough data for optimization
                equal_weight = 1.0 / len(available_assets)
                return {symbol: equal_weight for symbol in available_assets}
            
            # Create returns DataFrame
            min_length = min(len(returns) for returns in returns_data.values())
            returns_df = pd.DataFrame({
                symbol: returns[-min_length:] 
                for symbol, returns in returns_data.items()
            })
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Optimization objective: minimize risk for target return
            n_assets = len(returns_df.columns)
            
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}  # Target return
            ]
            
            # Bounds (no short selling, max 40% per asset)
            bounds = [(0, 0.4) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = dict(zip(returns_df.columns, result.x))
                logger.info(f"Portfolio optimization successful. Expected return: {target_return:.2%}")
                return optimal_weights
            else:
                logger.warning("Portfolio optimization failed, using equal weights")
                equal_weight = 1.0 / len(available_assets)
                return {symbol: equal_weight for symbol in available_assets}
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            equal_weight = 1.0 / len(available_assets)
            return {symbol: equal_weight for symbol in available_assets}
    
    def _calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        if len(returns) == 0:
            return 0
        var = self._calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)
    
    def _calculate_beta(self, portfolio_returns: np.ndarray, 
                      returns_df: pd.DataFrame) -> float:
        """Calculate portfolio beta (vs first asset as market proxy)"""
        if len(portfolio_returns) == 0 or returns_df.empty:
            return 1.0
        
        # Use first asset as market proxy
        market_returns = returns_df.iloc[:, 0].values
        if len(market_returns) == len(portfolio_returns):
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance > 0 else 1.0
        return 1.0
    
    def _calculate_correlation_risk(self, correlation_matrix: pd.DataFrame, 
                                  weights: np.ndarray) -> float:
        """Calculate correlation risk score"""
        if correlation_matrix.empty:
            return 0
        
        # Average correlation weighted by position sizes
        weighted_corr = 0
        total_weight = 0
        
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                corr = abs(correlation_matrix.iloc[i, j])
                weight = weights[i] * weights[j]
                weighted_corr += corr * weight
                total_weight += weight
        
        return weighted_corr / total_weight if total_weight > 0 else 0
    
    def _calculate_concentration_risk(self, weights: np.ndarray) -> float:
        """Calculate concentration risk using Herfindahl-Hirschman Index"""
        if len(weights) == 0:
            return 0
        return np.sum(weights ** 2)
    
    def _calculate_overall_risk_score(self, var: float, correlation_risk: float,
                                    concentration_risk: float, liquidity_risk: float,
                                    volatility: float, max_drawdown: float) -> float:
        """Calculate overall risk score (0-1, higher is riskier)"""
        # Normalize each risk component
        normalized_var = min(abs(var) / 0.1, 1.0)  # Normalize to 10% VaR
        normalized_correlation = min(correlation_risk, 1.0)
        normalized_concentration = min(concentration_risk, 1.0)
        normalized_liquidity = liquidity_risk
        normalized_volatility = min(volatility / 0.5, 1.0)  # Normalize to 50% volatility
        normalized_drawdown = min(abs(max_drawdown) / 0.2, 1.0)  # Normalize to 20% drawdown
        
        # Weighted average
        overall_score = (
            self.risk_weights['var'] * normalized_var +
            self.risk_weights['correlation'] * normalized_correlation +
            self.risk_weights['concentration'] * normalized_concentration +
            self.risk_weights['liquidity'] * normalized_liquidity +
            self.risk_weights['volatility'] * normalized_volatility +
            self.risk_weights['drawdown'] * normalized_drawdown
        )
        
        return min(overall_score, 1.0)
    
    def _calculate_position_beta(self, symbol: str, 
                               price_history: Dict[str, List[float]]) -> float:
        """Calculate beta for individual position"""
        if symbol not in price_history or len(price_history[symbol]) < 2:
            return 1.0
        
        # Use BTC as market proxy if available
        market_symbol = 'BTC/EUR'
        if market_symbol not in price_history:
            market_symbol = list(price_history.keys())[0]  # Use first available
        
        if market_symbol == symbol or market_symbol not in price_history:
            return 1.0
        
        # Calculate returns
        asset_prices = np.array(price_history[symbol])
        market_prices = np.array(price_history[market_symbol])
        
        min_length = min(len(asset_prices), len(market_prices))
        asset_returns = np.diff(asset_prices[-min_length:]) / asset_prices[-min_length:-1]
        market_returns = np.diff(market_prices[-min_length:]) / market_prices[-min_length:-1]
        
        if len(asset_returns) > 1 and len(market_returns) > 1:
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance > 0 else 1.0
        
        return 1.0
    
    def _calculate_position_correlation(self, symbol: str, 
                                      portfolio_positions: Dict[str, Any],
                                      price_history: Dict[str, List[float]]) -> float:
        """Calculate correlation between position and rest of portfolio"""
        if symbol not in price_history or len(price_history[symbol]) < 2:
            return 0
        
        # Calculate portfolio returns without this position
        other_positions = {k: v for k, v in portfolio_positions.items() if k != symbol}
        if not other_positions:
            return 0
        
        # Get returns for asset and portfolio
        asset_prices = np.array(price_history[symbol])
        asset_returns = np.diff(asset_prices) / asset_prices[:-1]
        
        # Calculate portfolio returns (simplified)
        portfolio_returns = []
        for other_symbol, position in other_positions.items():
            if other_symbol in price_history and len(price_history[other_symbol]) > 1:
                other_prices = np.array(price_history[other_symbol])
                other_returns = np.diff(other_prices) / other_prices[:-1]
                
                # Weight by position size
                weight = position.get('eur_value', 0)
                portfolio_returns.append(other_returns * weight)
        
        if not portfolio_returns:
            return 0
        
        # Average portfolio returns
        min_length = min(len(asset_returns), min(len(ret) for ret in portfolio_returns))
        portfolio_avg_returns = np.mean([ret[-min_length:] for ret in portfolio_returns], axis=0)
        
        # Calculate correlation
        if len(asset_returns) >= min_length and len(portfolio_avg_returns) >= min_length:
            correlation, _ = pearsonr(asset_returns[-min_length:], portfolio_avg_returns)
            return correlation if not np.isnan(correlation) else 0
        
        return 0
    
    def _calculate_optimal_position_size(self, symbol: str, 
                                       portfolio_positions: Dict[str, Any],
                                       price_history: Dict[str, List[float]]) -> float:
        """Calculate optimal position size based on risk"""
        # Kelly Criterion approximation
        if symbol not in price_history or len(price_history[symbol]) < 2:
            return 0
        
        prices = np.array(price_history[symbol])
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) == 0:
            return 0
        
        # Calculate Kelly fraction
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        if variance > 0:
            kelly_fraction = mean_return / variance
            # Cap at 25% for safety
            optimal_fraction = min(abs(kelly_fraction), 0.25)
        else:
            optimal_fraction = 0.05  # Default 5%
        
        # Adjust for correlation with existing positions
        correlation = self._calculate_position_correlation(symbol, portfolio_positions, price_history)
        correlation_adjustment = 1 - abs(correlation) * 0.5  # Reduce size if highly correlated
        
        return optimal_fraction * correlation_adjustment
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            portfolio_var=0, portfolio_cvar=0, max_drawdown=0,
            volatility=0, sharpe_ratio=0, beta=1.0,
            correlation_risk=0, concentration_risk=0, liquidity_risk=0.5,
            overall_risk_score=0.5
        )
    
    def _store_risk_metrics(self, metrics: RiskMetrics):
        """Store risk metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_metrics (
                    timestamp, portfolio_var, portfolio_cvar, max_drawdown,
                    volatility, sharpe_ratio, beta, correlation_risk,
                    concentration_risk, liquidity_risk, overall_risk_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metrics.portfolio_var, metrics.portfolio_cvar, metrics.max_drawdown,
                metrics.volatility, metrics.sharpe_ratio, metrics.beta,
                metrics.correlation_risk, metrics.concentration_risk,
                metrics.liquidity_risk, metrics.overall_risk_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing risk metrics: {e}")
    
    def _store_position_risk(self, position_risk: PositionRisk):
        """Store position risk in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO position_risk (
                    timestamp, symbol, position_size, value_at_risk, beta,
                    correlation_with_portfolio, liquidity_score, concentration_pct,
                    risk_contribution, recommended_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                position_risk.symbol, position_risk.position_size,
                position_risk.value_at_risk, position_risk.beta,
                position_risk.correlation_with_portfolio, position_risk.liquidity_score,
                position_risk.concentration_pct, position_risk.risk_contribution,
                position_risk.recommended_size
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing position risk: {e}")
    
    def _store_risk_alert(self, alert: RiskAlert):
        """Store risk alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_alerts (
                    timestamp, alert_type, severity, symbol, message,
                    current_value, threshold, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp, alert.alert_type, alert.severity,
                alert.symbol, alert.message, alert.current_value,
                alert.threshold, alert.recommendation
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing risk alert: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get latest risk metrics
            latest_metrics = pd.read_sql_query('''
                SELECT * FROM risk_metrics 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', conn)
            
            # Get recent alerts
            recent_alerts = pd.read_sql_query('''
                SELECT * FROM risk_alerts 
                WHERE timestamp >= datetime('now', '-1 day')
                ORDER BY timestamp DESC
            ''', conn)
            
            # Get position risks
            position_risks = pd.read_sql_query('''
                SELECT * FROM position_risk 
                WHERE timestamp >= datetime('now', '-1 day')
                ORDER BY timestamp DESC
            ''', conn)
            
            conn.close()
            
            return {
                'latest_metrics': latest_metrics.to_dict('records')[0] if not latest_metrics.empty else {},
                'recent_alerts': recent_alerts.to_dict('records'),
                'position_risks': position_risks.to_dict('records'),
                'risk_thresholds': self.risk_thresholds
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}
    
    def print_risk_report(self):
        """Print formatted risk report"""
        summary = self.get_risk_summary()
        
        print("\n" + "="*60)
        print("RISK MANAGEMENT REPORT")
        print("="*60)
        
        metrics = summary.get('latest_metrics', {})
        if metrics:
            print(f"Overall Risk Score:    {metrics.get('overall_risk_score', 0):.2f}")
            print(f"Portfolio VaR:         {metrics.get('portfolio_var', 0):.2%}")
            print(f"Max Drawdown:          {metrics.get('max_drawdown', 0):.2%}")
            print(f"Volatility:            {metrics.get('volatility', 0):.2%}")
            print(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Beta:                  {metrics.get('beta', 1):.2f}")
            print(f"Correlation Risk:      {metrics.get('correlation_risk', 0):.2f}")
            print(f"Concentration Risk:    {metrics.get('concentration_risk', 0):.2f}")
        
        alerts = summary.get('recent_alerts', [])
        if alerts:
            print("\n" + "-"*60)
            print("RECENT RISK ALERTS")
            print("-"*60)
            for alert in alerts[:5]:  # Show top 5
                severity_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}.get(alert.get('severity', 'MEDIUM'), "⚪")
                print(f"{severity_icon} {alert.get('alert_type', '')}: {alert.get('message', '')}")
        
        print("\n" + "="*60)

# Global risk manager instance
risk_manager = RiskManager() 