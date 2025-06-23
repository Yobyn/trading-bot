import ccxt
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from loguru import logger
from config import config

class TradingEngine:
    def __init__(self, exchange_name: str = None, api_key: str = None, secret: str = None):
        self.exchange_name = exchange_name or config.exchange
        self.api_key = api_key or config.exchange_api_key
        self.secret = secret or config.exchange_secret
        
        # Initialize exchange
        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.secret,
            'sandbox': config.paper_trading,
            'enableRateLimit': True,
        })
        
        # Portfolio state
        self.positions = {}
        self.orders = []
        self.portfolio_value = 10000.0  # Starting value
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Risk management
        self.max_positions = config.max_positions
        self.position_size = config.position_size
        self.stop_loss_pct = config.stop_loss_pct
        self.take_profit_pct = config.take_profit_pct
        self.max_daily_loss = config.max_daily_loss
        
    async def execute_trade(self, decision: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on LLM decision"""
        try:
            action = decision.get('action', 'HOLD')
            symbol = market_data.get('symbol', config.symbol)
            current_price = market_data.get('current_price', 0)
            
            if not config.trading_enabled:
                logger.info(f"Trading disabled. Would execute: {action} {symbol} at {current_price}")
                return self._create_paper_trade_result(action, symbol, current_price, decision)
            
            # Check risk management
            if not self._check_risk_limits():
                logger.warning("Risk limits exceeded, skipping trade")
                return {'status': 'rejected', 'reason': 'risk_limits_exceeded'}
            
            # Execute based on action
            if action == 'BUY':
                return await self._execute_buy(symbol, current_price, decision)
            elif action == 'SELL':
                return await self._execute_sell(symbol, current_price, decision)
            elif action == 'CLOSE':
                return await self._execute_close(symbol, current_price, decision)
            else:  # HOLD
                return {'status': 'no_action', 'action': 'HOLD', 'reason': decision.get('reason', '')}
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _execute_buy(self, symbol: str, price: float, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a buy order"""
        try:
            # Check if we already have a position
            if symbol in self.positions and self.positions[symbol]['side'] == 'long':
                logger.info(f"Already have long position in {symbol}")
                return {'status': 'no_action', 'action': 'HOLD', 'reason': 'Already have long position'}
            
            # Calculate position size
            position_value = self.portfolio_value * self.position_size
            quantity = position_value / price
            
            # Check if we have enough positions available
            if len(self.positions) >= self.max_positions:
                logger.warning(f"Maximum positions ({self.max_positions}) reached")
                return {'status': 'rejected', 'reason': 'max_positions_reached'}
            
            # Execute order
            if config.paper_trading:
                order_result = self._create_paper_order('buy', symbol, quantity, price)
            else:
                order_result = await self._place_real_order('buy', symbol, quantity, price)
            
            if order_result['status'] == 'success':
                # Update position
                self.positions[symbol] = {
                    'side': 'long',
                    'size': quantity,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'stop_loss': price * (1 - self.stop_loss_pct),
                    'take_profit': price * (1 + self.take_profit_pct)
                }
                
                logger.info(f"Opened long position: {symbol} {quantity} @ {price}")
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error executing buy: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _execute_sell(self, symbol: str, price: float, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a sell order (short position)"""
        try:
            # Check if we already have a short position
            if symbol in self.positions and self.positions[symbol]['side'] == 'short':
                logger.info(f"Already have short position in {symbol}")
                return {'status': 'no_action', 'action': 'HOLD', 'reason': 'Already have short position'}
            
            # Calculate position size
            position_value = self.portfolio_value * self.position_size
            quantity = position_value / price
            
            # Check if we have enough positions available
            if len(self.positions) >= self.max_positions:
                logger.warning(f"Maximum positions ({self.max_positions}) reached")
                return {'status': 'rejected', 'reason': 'max_positions_reached'}
            
            # Execute order
            if config.paper_trading:
                order_result = self._create_paper_order('sell', symbol, quantity, price)
            else:
                order_result = await self._place_real_order('sell', symbol, quantity, price)
            
            if order_result['status'] == 'success':
                # Update position
                self.positions[symbol] = {
                    'side': 'short',
                    'size': quantity,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'stop_loss': price * (1 + self.stop_loss_pct),
                    'take_profit': price * (1 - self.take_profit_pct)
                }
                
                logger.info(f"Opened short position: {symbol} {quantity} @ {price}")
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error executing sell: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _execute_close(self, symbol: str, price: float, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Close an existing position"""
        try:
            if symbol not in self.positions:
                logger.info(f"No position to close for {symbol}")
                return {'status': 'no_action', 'action': 'HOLD', 'reason': 'No position to close'}
            
            position = self.positions[symbol]
            quantity = position['size']
            
            # Execute order
            if config.paper_trading:
                order_result = self._create_paper_order('close', symbol, quantity, price, position)
            else:
                order_result = await self._place_real_order('close', symbol, quantity, price, position)
            
            if order_result['status'] == 'success':
                # Calculate P&L
                if position['side'] == 'long':
                    pnl = (price - position['entry_price']) * quantity
                else:  # short
                    pnl = (position['entry_price'] - price) * quantity
                
                # Update portfolio
                self.portfolio_value += pnl
                self.total_pnl += pnl
                self.daily_pnl += pnl
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"Closed position: {symbol} P&L: {pnl:.2f}")
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error executing close: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _place_real_order(self, side: str, symbol: str, quantity: float, price: float, position: Dict = None) -> Dict[str, Any]:
        """Place a real order on the exchange"""
        try:
            if side == 'close':
                # Close position - use opposite side of current position
                order_side = 'sell' if position['side'] == 'long' else 'buy'
            else:
                order_side = side
            
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=order_side,
                amount=quantity
            )
            
            return {
                'status': 'success',
                'order_id': order.get('id'),
                'side': order_side,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error placing real order: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _create_paper_order(self, side: str, symbol: str, quantity: float, price: float, position: Dict = None) -> Dict[str, Any]:
        """Create a paper trading order"""
        order_side = side
        if side == 'close':
            order_side = 'sell' if position['side'] == 'long' else 'buy'
        
        order = {
            'id': f"paper_{datetime.now().timestamp()}",
            'side': order_side,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().isoformat(),
            'paper_trading': True
        }
        
        self.orders.append(order)
        
        return {
            'status': 'success',
            'order_id': order['id'],
            'side': order_side,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': order['timestamp'],
            'paper_trading': True
        }
    
    def _create_paper_trade_result(self, action: str, symbol: str, price: float, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Create a paper trading result when trading is disabled"""
        return {
            'status': 'paper_trade',
            'action': action,
            'symbol': symbol,
            'price': price,
            'reason': decision.get('reason', ''),
            'timestamp': datetime.now().isoformat(),
            'paper_trading': True
        }
    
    def _check_risk_limits(self) -> bool:
        """Check if we're within risk limits"""
        # Check daily loss limit
        if self.daily_pnl < -(self.portfolio_value * self.max_daily_loss):
            logger.warning(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
            return False
        
        # Check portfolio risk
        total_exposure = sum(abs(pos['size'] * pos['entry_price']) for pos in self.positions.values())
        if total_exposure > self.portfolio_value * self.max_portfolio_risk:
            logger.warning(f"Portfolio risk limit exceeded: {total_exposure:.2f}")
            return False
        
        return True
    
    async def check_stop_losses(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check and execute stop losses"""
        closed_positions = []
        current_price = market_data.get('current_price', 0)
        
        for symbol, position in list(self.positions.items()):
            should_close = False
            reason = ""
            
            # Check stop loss
            if position['side'] == 'long' and current_price <= position['stop_loss']:
                should_close = True
                reason = "stop_loss"
            elif position['side'] == 'short' and current_price >= position['stop_loss']:
                should_close = True
                reason = "stop_loss"
            
            # Check take profit
            elif position['side'] == 'long' and current_price >= position['take_profit']:
                should_close = True
                reason = "take_profit"
            elif position['side'] == 'short' and current_price <= position['take_profit']:
                should_close = True
                reason = "take_profit"
            
            if should_close:
                result = await self._execute_close(symbol, current_price, {'action': 'CLOSE', 'reason': reason})
                if result['status'] == 'success':
                    closed_positions.append({
                        'symbol': symbol,
                        'reason': reason,
                        'pnl': result.get('pnl', 0)
                    })
        
        return closed_positions
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        total_exposure = sum(abs(pos['size'] * pos['entry_price']) for pos in self.positions.values())
        
        return {
            'portfolio_value': self.portfolio_value,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'total_exposure': total_exposure,
            'position_count': len(self.positions),
            'positions': self.positions,
            'orders': self.orders[-10:],  # Last 10 orders
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_daily_pnl(self):
        """Reset daily P&L (call this daily)"""
        self.daily_pnl = 0.0 