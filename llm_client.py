import json
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger
from config import config

class LLMClient:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url: str = base_url or getattr(config, 'llm_base_url', None) or "http://localhost:11434"
        self.model: str = model or getattr(config, 'llm_model', None) or "llama2"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from Ollama"""
        if not self.session:
            raise Exception("LLM client session not initialized. Use 'async with' context manager.")
            
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            logger.info(f"Sending request to Ollama at {url}")
            
            # Format payload for human-readable logging
            logger.info("📤 Request Payload:")
            logger.info(f"  Model: {payload.get('model', '')}")
            logger.info(f"  Stream: {payload.get('stream', False)}")
            
            if "system" in payload:
                logger.info("  System Prompt:")
                for line in payload["system"].split('\n'):
                    logger.info(f"    {line}")
                    
            if "prompt" in payload:
                logger.info("  User Prompt:")
                for line in payload["prompt"].split('\n'):
                    logger.info(f"    {line}")
            
            async with self.session.post(url, json=payload, timeout=60) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get("response", "").strip()
                    logger.info(f"📥 Ollama Response:\n{response_text}")
                    return response_text
                else:
                    logger.error(f"Ollama API returned status {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response body: {response_text}")
                    raise Exception(f"Ollama API error: {response.status}")
                    
        except asyncio.TimeoutError:
            logger.error("Ollama request timed out after 60 seconds")
            raise Exception("Ollama request timed out")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            raise Exception(f"Could not connect to Ollama: {e}")
    
    async def get_asset_selection(self, assets: List[Dict[str, Any]], portfolio_value: float, available_capital: float) -> Dict[str, Any]:
        """Get asset selection from LLM based on multiple asset data"""
        system_prompt = """You are an expert crypto trading bot. Analyze the available assets and select the best one to invest in.
        
        Consider:
        - Current price vs weekly average (discount opportunities)
        - Market trends and momentum
        - Risk/reward potential
        - Portfolio diversification
        
        Respond with ONLY the asset symbol (e.g., BTC/EUR, ETH/EUR, SOL/EUR, ADA/EUR) and a brief reason.
        Keep responses concise and actionable."""
        
        # Format asset data for LLM analysis
        asset_analysis = []
        for asset in assets:
            symbol = asset.get('symbol', 'Unknown')
            current_price = asset.get('current_price', 0)
            weekly_avg = asset.get('weekly_average', 0)
            
            # Calculate discount/premium
            if weekly_avg > 0:
                discount_pct = ((weekly_avg - current_price) / weekly_avg) * 100
                if discount_pct > 0:
                    discount_desc = f"{discount_pct:.1f}% discount"
                else:
                    discount_desc = f"{abs(discount_pct):.1f}% premium"
            else:
                discount_desc = "No historical data"
            
            asset_analysis.append(f"""
{symbol}:
  Current Price: €{current_price:.6f}
  Weekly Average: €{weekly_avg:.6f}
  Valuation: {discount_desc}
  Bid: €{asset.get('bid', 0):.6f}
  Ask: €{asset.get('ask', 0):.6f}""")
        
        prompt = f"""Asset Selection Analysis:

Available Assets:
{''.join(asset_analysis)}

Portfolio Status:
  Total Value: €{portfolio_value:.2f}
  Available Capital: €{available_capital:.2f}

Select the best asset to invest all €{available_capital:.2f} in. Consider:
- Which asset offers the best value (biggest discount from weekly average)?
- Which asset has the most potential for growth?
- Which asset fits best with current market conditions?

Respond with the asset symbol and your reasoning."""
        
        logger.info("Requesting asset selection from Ollama")
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"User prompt: {prompt}")
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Parse the response to extract asset symbol
        # Look for exact asset symbol matches at the beginning of lines or after common phrases
        import re
        
        selected_symbol = None
        
        # First, try to find explicit asset mentions at start of response or after key phrases
        response_upper = response.upper()
        
        # Look for patterns like "APE/EUR", "BTC/EUR" etc. at start or after keywords
        for asset in assets:
            symbol = asset.get('symbol', '')
            symbol_upper = symbol.upper()
            
            # Check if symbol appears at start of response
            if response_upper.startswith(symbol_upper):
                selected_symbol = symbol
                break
                
            # Check for patterns like "I recommend [SYMBOL]" or "investing in [SYMBOL]"
            patterns = [
                rf'\b(?:recommend|investing in|select|choose)\s+{re.escape(symbol_upper)}\b',
                rf'\b{re.escape(symbol_upper)}\b(?:\s+(?:is|offers|has))',
                rf'^{re.escape(symbol_upper)}\b',  # At start of line
            ]
            
            for pattern in patterns:
                if re.search(pattern, response_upper, re.MULTILINE):
                    selected_symbol = symbol
                    break
            
            if selected_symbol:
                break
        
        # If no explicit symbol found, look for mentions of the base currency (like "APE", "BTC")
        if not selected_symbol:
            for asset in assets:
                symbol = asset.get('symbol', '')
                base_currency = symbol.split('/')[0] if '/' in symbol else symbol
                base_upper = base_currency.upper()
                
                # Look for base currency mentions with investment context
                patterns = [
                    rf'\b(?:recommend|investing in|select|choose)\s+{re.escape(base_upper)}\b',
                    rf'\b{re.escape(base_upper)}\s+(?:\(|offers|has|is)',
                    rf'^\s*{re.escape(base_upper)}\b',  # At start of line
                ]
                
                for pattern in patterns:
                    if re.search(pattern, response_upper, re.MULTILINE):
                        selected_symbol = symbol
                        break
                
                if selected_symbol:
                    break
        
        # Default to first asset if no clear selection
        if not selected_symbol and assets:
            selected_symbol = assets[0].get('symbol', '')
        
        decision = {
            "symbol": selected_symbol,
            "reason": response,
            "confidence": 0.8,
            "timestamp": assets[0].get('timestamp') if assets else None
        }
        
        # Format decision for human-readable logging
        logger.info("🤖 Asset Selection:")
        logger.info(f"  Selected: {selected_symbol}")
        logger.info(f"  Confidence: {decision['confidence']*100:.0f}%")
        logger.info(f"  Reason: {response}")
        
        return decision

    async def get_trading_decision(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get trading decision from LLM based on market data"""
        system_prompt = """You are an expert trading bot. Analyze the market data and provide a trading decision.
        
        Available actions:
        - BUY: Enter a long position
        - SELL: Enter a short position  
        - HOLD: No action, maintain current position
        - CLOSE: Close current position
        
        Respond with ONLY the action (BUY/SELL/HOLD/CLOSE) and optionally a brief reason.
        Keep responses concise and actionable."""
        
        # Check if this is a sell decision with profit/loss data
        buy_price = market_data.get('buy_price')
        profit_loss_pct = market_data.get('profit_loss_pct')
        profit_loss_eur = market_data.get('profit_loss_eur')
        
        prompt = f"""Market Data Analysis:
        
        Symbol: {market_data.get('symbol', 'Unknown')}
        Current Price: {market_data.get('current_price', 'Unknown')}"""
        
        # Add buy price and profit/loss info if available
        if buy_price is not None:
            prompt += f"""
        Original Buy Price: {buy_price}
        Profit/Loss: {profit_loss_pct:+.2f}% (€{profit_loss_eur:+.2f})"""
        
        prompt += f"""
        Price vs Yearly Avg: {market_data.get('price_vs_yearly_avg_pct', 'Unknown')}%
        Volume 24h: {market_data.get('volume_24h', 'Unknown')}
        Yearly Average: {market_data.get('yearly_average', 'Unknown')}
        RSI: {market_data.get('rsi', 'Unknown')}
        MACD: {market_data.get('macd', 'Unknown')}
        Moving Average (20): {market_data.get('ma_20', 'Unknown')}
        Moving Average (50): {market_data.get('ma_50', 'Unknown')}
        
        Current Position: {market_data.get('current_position', 'None')}
        Portfolio Value: {market_data.get('portfolio_value', 'Unknown')}
        
        Based on this data, what should I do? Respond with only the action and brief reason."""
        
        logger.info("Requesting trading decision from Ollama")
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"User prompt: {prompt}")
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Parse the response - look for explicit action keywords
        action = "HOLD"  # Default action
        reason = response
        
        if response:
            response_upper = response.upper()
            
            # Look for explicit action statements in order of priority
            if "ACTION: HOLD" in response_upper or "RECOMMEND HOLD" in response_upper or response_upper.startswith("HOLD"):
                action = "HOLD"
            elif "ACTION: BUY" in response_upper or "RECOMMEND BUY" in response_upper or response_upper.startswith("BUY"):
                action = "BUY"
            elif "ACTION: SELL" in response_upper or "RECOMMEND SELL" in response_upper or response_upper.startswith("SELL"):
                action = "SELL"
            elif "ACTION: CLOSE" in response_upper or "RECOMMEND CLOSE" in response_upper or response_upper.startswith("CLOSE"):
                action = "CLOSE"
            # Fall back to keyword search if no explicit action found
            elif "HOLDING" in response_upper or "HOLD" in response_upper:
                action = "HOLD"
            elif "BUYING" in response_upper or "BUY" in response_upper:
                action = "BUY"
            elif "SELLING" in response_upper or "SELL" in response_upper:
                action = "SELL"
            elif "CLOSING" in response_upper or "CLOSE" in response_upper:
                action = "CLOSE"
        
        decision = {
            "action": action,
            "reason": reason,
            "confidence": 0.8,
            "timestamp": market_data.get("timestamp")
        }
        
        # Format decision for human-readable logging
        logger.info("🤖 Trading Decision:")
        logger.info(f"  Action: {action}")
        logger.info(f"  Confidence: {decision['confidence']*100:.0f}%")
        logger.info(f"  Reason: {reason}")
        
        return decision 