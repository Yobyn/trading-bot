import json
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from aiohttp import ClientTimeout
from loguru import logger
from config import config
from audit_trail import audit_trail

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
            try:
                await self.session.close()
            except (asyncio.CancelledError, Exception) as e:
                # Ignore cleanup errors during shutdown
                logger.debug(f"Session cleanup error (expected during shutdown): {e}")
                pass
    
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
            
            async with self.session.post(url, json=payload, timeout=ClientTimeout(total=60)) as response:
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
        except asyncio.CancelledError:
            logger.debug("Ollama request cancelled (shutdown in progress)")
            raise  # Re-raise CancelledError to be handled by caller
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
        
        # Debug logging to see what we're working with
        logger.debug(f"🔍 Parsing asset selection from response: {response_upper[:100]}...")
        
        # Look for patterns like "APE/EUR", "BTC/EUR" etc. at start or after keywords
        for asset in assets:
            symbol = asset.get('symbol', '')
            symbol_upper = symbol.upper()
            base_currency = symbol.split('/')[0] if '/' in symbol else symbol
            base_upper = base_currency.upper()
            
            logger.debug(f"🔍 Checking patterns for {symbol} (base: {base_currency})")
            
            # Check if full symbol appears at start of response
            if response_upper.startswith(symbol_upper):
                logger.debug(f"✅ Found {symbol} at start of response")
                selected_symbol = symbol
                break
                
            # More flexible patterns to catch various recommendation formats
            patterns = [
                # Direct symbol mention in investment context
                rf'\b(?:recommend|select|choose)\s+(?:investing\s+in\s+)?{re.escape(symbol_upper)}\b',
                rf'\b(?:investing\s+in|invest\s+in)\s+{re.escape(symbol_upper)}\b',
                rf'\b(?:recommend|select|choose)\s+{re.escape(symbol_upper)}\b',
                rf'\b{re.escape(symbol_upper)}\b(?:\s+(?:is|offers|has))',
                rf'^{re.escape(symbol_upper)}\b',  # At start of line
                # Handle "recommend investing in SYMBOL" pattern
                rf'\brecommend\s+investing\s+in\s+{re.escape(symbol_upper)}\b',
                rf'\bwould\s+recommend\s+investing\s+in\s+{re.escape(symbol_upper)}\b',
                # Base currency patterns - THIS IS CRUCIAL
                rf'\b(?:recommend|select|choose)\s+(?:investing\s+in\s+)?{re.escape(base_upper)}\b',
                rf'\b(?:investing\s+in|invest\s+in)\s+{re.escape(base_upper)}\b',
                rf'\brecommend\s+investing\s+in\s+{re.escape(base_upper)}\b',
                rf'\bwould\s+recommend\s+investing\s+in\s+{re.escape(base_upper)}\b',
                rf'\binvest\s+in\s+(?:is\s+)?{re.escape(base_upper)}\b',
                rf'\bbest\s+asset\s+to\s+invest\s+in\s+is\s+{re.escape(base_upper)}\b',
                # NEW PATTERNS to handle common LLM phrasings
                rf'\bthe\s+best\s+asset\s+to\s+invest\s+in\s+is\s+{re.escape(symbol_upper)}[.,!]?',
                rf'\bthe\s+best\s+asset\s+to\s+invest\s+in\s+is\s+{re.escape(base_upper)}[.,!]?',
                rf'\binvest\s+all\s+.*\s+in\s+{re.escape(symbol_upper)}[.,!]?',
                rf'\binvest\s+all\s+.*\s+in\s+{re.escape(base_upper)}[.,!]?',
                rf'\brecommend\s+investing\s+all\s+.*\s+in\s+{re.escape(symbol_upper)}[.,!]?',
                rf'\brecommend\s+investing\s+all\s+.*\s+in\s+{re.escape(base_upper)}[.,!]?',
                rf'\bi\s+recommend\s+investing\s+.*\s+in\s+{re.escape(symbol_upper)}[.,!]?',
                rf'\bi\s+recommend\s+investing\s+.*\s+in\s+{re.escape(base_upper)}[.,!]?',
                rf'\btherefore[,\s]*\s*i\s+recommend\s+investing\s+.*\s+in\s+{re.escape(symbol_upper)}[.,!]?',
                rf'\btherefore[,\s]*\s*i\s+recommend\s+investing\s+.*\s+in\s+{re.escape(base_upper)}[.,!]?',
                # SIMPLIFIED "would recommend" patterns that actually work
                rf'WOULD\s+RECOMMEND\s+INVESTING\s+IN\s+{re.escape(symbol_upper)}',
                rf'WOULD\s+RECOMMEND\s+INVESTING\s+IN\s+{re.escape(base_upper)}',
                rf'I\s+WOULD\s+RECOMMEND\s+INVESTING\s+IN\s+{re.escape(symbol_upper)}',
                rf'I\s+WOULD\s+RECOMMEND\s+INVESTING\s+IN\s+{re.escape(base_upper)}',
                rf'RECOMMEND\s+INVESTING\s+IN\s+{re.escape(symbol_upper)}',
                rf'RECOMMEND\s+INVESTING\s+IN\s+{re.escape(base_upper)}',
                rf'INVESTING\s+IN\s+{re.escape(symbol_upper)}',
                rf'INVESTING\s+IN\s+{re.escape(base_upper)}',
                # Handle punctuation variations
                rf'\b{re.escape(symbol_upper)}[.,!]?\s*$',  # Symbol at end of sentence
                rf'\b{re.escape(base_upper)}[.,!]?\s*$',    # Base currency at end of sentence
            ]
            
            for i, pattern in enumerate(patterns):
                if re.search(pattern, response_upper, re.MULTILINE):
                    logger.debug(f"✅ Found {symbol} using pattern {i}: {pattern}")
                    selected_symbol = symbol
                    break
            
            if selected_symbol:
                break

        # If no explicit symbol found, do a second pass looking for standalone base currencies
        if not selected_symbol:
            logger.debug("🔍 Second pass: looking for standalone base currencies")
            for asset in assets:
                symbol = asset.get('symbol', '')
                base_currency = symbol.split('/')[0] if '/' in symbol else symbol
                base_upper = base_currency.upper()
                
                logger.debug(f"🔍 Second pass checking {base_currency}")
                
                # Look for base currency mentions with investment context
                patterns = [
                    rf'\b(?:recommend|investing in|select|choose)\s+{re.escape(base_upper)}\b',
                    rf'\b{re.escape(base_upper)}\s+(?:\(|offers|has|is)',
                    rf'^\s*{re.escape(base_upper)}\b',  # At start of line
                    rf'\binvest\s+in\s+is\s+{re.escape(base_upper)}\b',  # "invest in is BAT"
                    rf'\basset\s+to\s+invest\s+in\s+is\s+{re.escape(base_upper)}\b',  # "asset to invest in is BAT"
                    # NEW COMPREHENSIVE PATTERNS for second pass
                    rf'\bthe\s+best\s+asset\s+to\s+invest\s+in\s+is\s+{re.escape(base_upper)}[.,!]?',
                    rf'\binvest\s+all\s+.*\s+in\s+{re.escape(base_upper)}[.,!]?',
                    rf'\brecommend\s+investing\s+all\s+.*\s+in\s+{re.escape(base_upper)}[.,!]?',
                    rf'\bi\s+recommend\s+investing\s+.*\s+in\s+{re.escape(base_upper)}[.,!]?',
                    rf'\btherefore[,\s]*\s*i\s+recommend\s+investing\s+.*\s+in\s+{re.escape(base_upper)}[.,!]?',
                    rf'\b{re.escape(base_upper)}[.,!]?\s*$',  # Base currency at end of sentence
                    # SIMPLIFIED "would recommend" patterns for second pass
                    rf'WOULD\s+RECOMMEND\s+INVESTING\s+IN\s+{re.escape(base_upper)}',
                    rf'I\s+WOULD\s+RECOMMEND\s+INVESTING\s+IN\s+{re.escape(base_upper)}',
                    rf'RECOMMEND\s+INVESTING\s+IN\s+{re.escape(base_upper)}',
                    rf'INVESTING\s+IN\s+{re.escape(base_upper)}',
                ]
                
                for i, pattern in enumerate(patterns):
                    if re.search(pattern, response_upper, re.MULTILINE):
                        logger.debug(f"✅ Second pass found {symbol} using pattern {i}: {pattern}")
                        selected_symbol = symbol
                        break
                
                if selected_symbol:
                    break

        # Log the final result
        if selected_symbol:
            logger.debug(f"🎯 Final selection: {selected_symbol}")
        else:
            logger.debug("❌ No asset matched, will default to first asset")
            
        # Default to first asset if no clear selection
        if not selected_symbol and assets:
            selected_symbol = assets[0].get('symbol', '')
            logger.debug(f"🔄 Defaulted to first asset: {selected_symbol}")
        
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
        
        # Get context information
        trading_phase = market_data.get('trading_phase', 'INVESTMENT')
        current_position = market_data.get('current_position', {})
        has_position = current_position.get('has_position', False)
        
        # Build context-aware system prompt with improved trading logic
        if trading_phase == "INVESTMENT" and not has_position:
            # Investment phase, no current position - can only BUY or HOLD
            available_actions = """Available actions:
        - BUY: Enter a long position with available cash (when price is below averages or RSI is oversold)
        - HOLD: Skip this asset and look for better opportunities (when price is overvalued or RSI is overbought)"""
            context_instruction = "INVESTMENT PHASE: Look for undervalued assets to buy. Prefer assets trading below their 3-month average or with oversold RSI (<30). Avoid overvalued assets trading above averages with overbought RSI (>70)."
            
        elif trading_phase == "INVESTMENT" and has_position:
            # Investment phase, has position - can add more, hold, or reduce
            available_actions = """Available actions:
        - BUY: Add more to the existing position (when price is still undervalued)
        - HOLD: Maintain current position size (when price is at fair value)
        - SELL: Reduce position size partially (when price is overvalued or RSI overbought)
        - CLOSE: Close entire position (when technical indicators strongly suggest reversal)"""
            context_instruction = "INVESTMENT PHASE: You have available cash and an existing position. Consider whether to add more (if still undervalued), maintain (if at fair value), or reduce (if overvalued)."
            
        elif trading_phase == "MANAGEMENT" and has_position:
            # Management phase, has position - focus on optimal portfolio management
            available_actions = """Available actions:
        - HOLD: Keep the current position (when technical indicators are neutral or favorable)
        - SELL: Liquidate this position (when overbought, overvalued, or showing reversal signals)"""
            context_instruction = "MANAGEMENT PHASE: Optimize portfolio by selling positions that are overbought (RSI >70), overvalued (above averages), or showing technical weakness. Hold positions that are still technically sound or undervalued."
            
        else:
            # Fallback - shouldn't happen in normal operation
            available_actions = """Available actions:
        - HOLD: No action"""
            context_instruction = "No clear trading context. Default to HOLD for safety."
        
        system_prompt = f"""You are an expert trading bot. Analyze the market data and provide a trading decision based on technical analysis and price valuation.
        
        {available_actions}
        
        TRADING CONTEXT: {context_instruction}
        
        KEY DECISION FACTORS:
        - SELL when: RSI >70 (overbought), price significantly above averages, or showing reversal signals
        - BUY when: RSI <30 (oversold), price below averages (discount), or showing strong momentum
        - HOLD when: RSI 30-70 (neutral), price near averages, or unclear signals
        
        Respond with ONLY the action and optionally a brief reason.
        Keep responses concise and actionable."""
        
        # Check if this is a sell decision with profit/loss data
        buy_price = market_data.get('buy_price')
        profit_loss_pct = market_data.get('profit_loss_pct')
        profit_loss_eur = market_data.get('profit_loss_eur')
        
        # Enhanced price vs average analysis with explicit comparisons
        current_price = market_data.get('current_price', 0)
        three_month_avg = market_data.get('three_month_average', 0)
        weekly_avg = market_data.get('weekly_average', 0)
        
        # Ensure values are numeric (handle cases where they might be strings)
        try:
            if current_price is None or current_price in ['Unknown', '']:
                current_price = 0
            else:
                current_price = float(current_price)
                
            if three_month_avg is None or three_month_avg in ['Unknown', '']:
                three_month_avg = 0
            else:
                three_month_avg = float(three_month_avg)
                
            if weekly_avg is None or weekly_avg in ['Unknown', '']:
                weekly_avg = 0
            else:
                weekly_avg = float(weekly_avg)
        except (ValueError, TypeError):
            current_price = 0
            three_month_avg = 0
            weekly_avg = 0
        
        # Format averages properly (avoid scientific notation)
        three_month_avg_str = f"€{three_month_avg:.6f}" if three_month_avg > 0 else "Unknown"
        weekly_avg_str = f"€{weekly_avg:.6f}" if weekly_avg > 0 else "Unknown"
        
        # Calculate and format explicit price comparisons
        price_vs_3month = ""
        price_vs_weekly = ""
        
        if three_month_avg > 0:
            diff_3month_pct = ((current_price - three_month_avg) / three_month_avg) * 100
            if diff_3month_pct > 0:
                price_vs_3month = f"ABOVE 3-month avg by {diff_3month_pct:.1f}% (premium)"
            else:
                price_vs_3month = f"BELOW 3-month avg by {abs(diff_3month_pct):.1f}% (discount)"
        else:
            price_vs_3month = "No 3-month data available"
            
        if weekly_avg > 0:
            diff_weekly_pct = ((current_price - weekly_avg) / weekly_avg) * 100
            if diff_weekly_pct > 0:
                price_vs_weekly = f"ABOVE weekly avg by {diff_weekly_pct:.1f}% (premium)"
            else:
                price_vs_weekly = f"BELOW weekly avg by {abs(diff_weekly_pct):.1f}% (discount)"
        else:
            price_vs_weekly = "No weekly data available"
        
        # Get RSI for decision guidance
        rsi_raw = market_data.get('rsi', 50)
        try:
            rsi = float(rsi_raw) if rsi_raw not in [None, 'Unknown', ''] else 50.0
        except (ValueError, TypeError):
            rsi = 50.0
        
        # Add RSI interpretation
        rsi_status = ""
        if rsi > 70:
            rsi_status = "OVERBOUGHT - Consider selling"
        elif rsi < 30:
            rsi_status = "OVERSOLD - Consider buying"
        else:
            rsi_status = "NEUTRAL"
        
        prompt = f"""Market Data Analysis:
        
        Symbol: {market_data.get('symbol', 'Unknown')}
        Current Price: €{current_price:.6f}"""
        
        # Add buy price and profit/loss info if available
        if buy_price is not None:
            try:
                buy_price_float = float(buy_price) if buy_price is not None and buy_price not in ['Unknown', ''] else 0.0
                profit_loss_pct_float = float(profit_loss_pct) if profit_loss_pct is not None and profit_loss_pct not in ['Unknown', ''] else 0.0
                profit_loss_eur_float = float(profit_loss_eur) if profit_loss_eur is not None and profit_loss_eur not in ['Unknown', ''] else 0.0
                
                prompt += f"""
        Original Buy Price: €{buy_price_float:.6f}
        Profit/Loss: {profit_loss_pct_float:+.2f}% (€{profit_loss_eur_float:+.2f})"""
            except (ValueError, TypeError):
                prompt += f"""
        Original Buy Price: {buy_price}
        Profit/Loss: {profit_loss_pct}% (€{profit_loss_eur})"""
        
        # Handle portfolio value safely
        portfolio_value = market_data.get('portfolio_value', 'Unknown')
        try:
            portfolio_value_float = float(portfolio_value) if portfolio_value not in [None, 'Unknown', ''] else 0
            portfolio_value_str = f"€{portfolio_value_float:.2f}"
        except (ValueError, TypeError):
            portfolio_value_str = str(portfolio_value)
        
        prompt += f"""
        
        PRICE ANALYSIS:
        - 3-Month Average: {three_month_avg_str}
        - Weekly Average: {weekly_avg_str}
        - Current vs 3-Month: {price_vs_3month}
        - Current vs Weekly: {price_vs_weekly}
        
        TECHNICAL INDICATORS:
        - RSI: {rsi:.1f} ({rsi_status})
        - MACD: {market_data.get('macd', 'Unknown')}
        - Volume 24h: {market_data.get('volume_24h', 'Unknown')}
        - Moving Average (20): {market_data.get('ma_20', 'Unknown')}
        - Moving Average (50): {market_data.get('ma_50', 'Unknown')}
        
        PORTFOLIO INFO:
        - Current Position: {market_data.get('current_position', 'None')}
        - Portfolio Value: {portfolio_value_str}"""
        
        # Add phase instruction if available
        if market_data.get('phase_instruction'):
            prompt += f"""
        
        TRADING CONTEXT: {market_data.get('phase_instruction')}"""
        
        prompt += """
        
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
        
        # AUDIT TRAIL: Log the complete LLM interaction
        try:
            symbol = market_data.get('symbol', 'Unknown')
            trading_phase = market_data.get('trading_phase', 'Unknown')
            audit_trail.log_llm_interaction(
                symbol=symbol,
                system_prompt=system_prompt,
                user_prompt=prompt,
                llm_response=response,
                parsed_decision=decision,
                market_data=market_data,
                trading_phase=trading_phase
            )
        except Exception as e:
            logger.warning(f"Failed to log LLM interaction to audit trail: {e}")
        
        # Format decision for human-readable logging
        logger.info("🤖 Trading Decision:")
        logger.info(f"  Action: {action}")
        logger.info(f"  Confidence: {decision['confidence']*100:.0f}%")
        logger.info(f"  Reason: {reason}")
        
        return decision

    async def get_top_crypto_recommendations(self, assets: List[Dict[str, Any]], total_capital: float) -> Dict[str, Any]:
        """Get LLM's top 5 cryptocurrency recommendations for portfolio rebalancing"""
        system_prompt = """You are an expert cryptocurrency portfolio manager. Analyze all available cryptocurrencies and select the top 5 for a balanced, diversified portfolio.
        
        Consider:
        - Market trends and technical indicators
        - Price relative to historical averages (discount/premium)
        - Risk diversification across different crypto categories
        - Growth potential and market fundamentals
        - Current market conditions
        
        Respond with EXACTLY 5 cryptocurrency symbols and brief reasoning for each."""
        
        # Prepare asset analysis for LLM
        asset_analysis = []
        for asset in assets:
            symbol = asset.get('symbol', '')
            current_price = asset.get('current_price', 0)
            three_month_avg = asset.get('three_month_average', 0)
            
            # Calculate discount/premium vs 3-month average
            if three_month_avg > 0:
                discount_pct = ((three_month_avg - current_price) / three_month_avg) * 100
                if discount_pct > 0:
                    valuation = f"{discount_pct:.1f}% discount"
                else:
                    valuation = f"{abs(discount_pct):.1f}% premium"
            else:
                valuation = "No historical data"
            
            # Get technical indicators
            rsi = asset.get('rsi', 50)
            price_vs_yearly = asset.get('price_vs_yearly_avg_pct', 0)
            
            asset_analysis.append(f"""
{symbol}:
  Current: €{current_price:.2f}
  3-Month Avg: €{three_month_avg:.2f}
  Valuation: {valuation}
  RSI: {rsi:.1f}
  vs Yearly: {price_vs_yearly:+.1f}%""")
        
        prompt = f"""Portfolio Rebalancing Analysis:

Total Capital: €{total_capital:.2f}

Available Cryptocurrencies:
{''.join(asset_analysis)}

Select the TOP 5 cryptocurrencies for a balanced portfolio. Consider:
- Diversification across different crypto types (major coins, DeFi, Layer 1s, etc.)
- Current valuations (prefer discounted assets)
- Technical indicators and market momentum
- Risk management and growth potential

Respond with EXACTLY 5 symbols in order of preference, with brief reasoning for each."""
        
        logger.info("Requesting top 5 crypto recommendations from LLM")
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"User prompt: {prompt}")
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Parse response to extract the 5 recommended symbols
        import re
        selected_symbols = []
        
        # Look for asset symbols in the response
        response_upper = response.upper()
        
        for asset in assets:
            symbol = asset.get('symbol', '')
            symbol_upper = symbol.upper()
            base_currency = symbol.split('/')[0] if '/' in symbol else symbol
            base_upper = base_currency.upper()
            
            # Check for symbol or base currency mentions
            if symbol_upper in response_upper or base_upper in response_upper:
                if symbol not in selected_symbols:
                    selected_symbols.append(symbol)
                    if len(selected_symbols) >= 5:
                        break
        
        # If we didn't find 5, add top assets by market cap as fallback
        if len(selected_symbols) < 5:
            major_cryptos = ['BTC/EUR', 'ETH/EUR', 'SOL/EUR', 'ADA/EUR', 'XRP/EUR']
            for crypto in major_cryptos:
                if crypto not in selected_symbols and any(a.get('symbol') == crypto for a in assets):
                    selected_symbols.append(crypto)
                    if len(selected_symbols) >= 5:
                        break
        
        # Ensure we have exactly 5
        selected_symbols = selected_symbols[:5]
        
        decision = {
            "recommended_cryptos": selected_symbols,
            "reasoning": response,
            "total_capital": total_capital,
            "timestamp": assets[0].get('timestamp') if assets else None
        }
        
        # Format decision for human-readable logging
        logger.info("🤖 Top 5 Crypto Recommendations:")
        for i, symbol in enumerate(selected_symbols, 1):
            logger.info(f"  {i}. {symbol}")
        logger.info(f"  Reasoning: {response}")
        
        return decision

    async def get_portfolio_allocation(self, recommended_cryptos: List[str], total_capital: float, crypto_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get LLM's recommended allocation percentages for the selected cryptocurrencies"""
        system_prompt = """You are an expert portfolio manager. Given 5 selected cryptocurrencies and total capital, determine the optimal allocation percentage for each.
        
        Consider:
        - Risk levels of each cryptocurrency
        - Current market conditions and valuations
        - Diversification principles
        - Growth potential vs stability
        
        Respond with EXACT percentages that sum to 100%. Format as:
        SYMBOL: XX%
        
        Be specific with numbers - avoid ranges."""
        
        # Prepare data for the selected cryptos
        crypto_analysis = []
        for symbol in recommended_cryptos:
            # Find the crypto data
            crypto_info = next((c for c in crypto_data if c.get('symbol') == symbol), None)
            if crypto_info:
                current_price = crypto_info.get('current_price', 0)
                three_month_avg = crypto_info.get('three_month_average', 0)
                price_vs_yearly = crypto_info.get('price_vs_yearly_avg_pct', 0)
                rsi = crypto_info.get('rsi', 50)
                
                # Calculate discount/premium
                if three_month_avg > 0:
                    discount_pct = ((three_month_avg - current_price) / three_month_avg) * 100
                    if discount_pct > 0:
                        valuation = f"{discount_pct:.1f}% discount"
                    else:
                        valuation = f"{abs(discount_pct):.1f}% premium"
                else:
                    valuation = "No historical data"
                
                crypto_analysis.append(f"""
{symbol}:
  Current: €{current_price:.2f}
  Valuation: {valuation}
  vs Yearly: {price_vs_yearly:+.1f}%
  RSI: {rsi:.1f}""")
        
        prompt = f"""Portfolio Allocation Decision:

Total Capital: €{total_capital:.2f}
Selected Cryptocurrencies:
{''.join(crypto_analysis)}

Determine the optimal allocation percentage for each cryptocurrency.
Consider:
- Lower allocation for higher-risk assets
- Higher allocation for undervalued/discounted assets
- Balance between growth potential and stability
- Risk management across the portfolio

Respond with EXACT percentages that sum to 100%:
{recommended_cryptos[0]}: XX%
{recommended_cryptos[1]}: XX%
{recommended_cryptos[2]}: XX%
{recommended_cryptos[3]}: XX%
{recommended_cryptos[4]}: XX%

Include brief reasoning for the allocation strategy."""
        
        logger.info("Requesting portfolio allocation from LLM")
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"User prompt: {prompt}")
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Parse response to extract allocation percentages
        import re
        allocations = {}
        
        # Look for patterns like "SYMBOL: XX%" in the response
        for symbol in recommended_cryptos:
            base_currency = symbol.split('/')[0] if '/' in symbol else symbol
            
            # Try different patterns to find the allocation
            patterns = [
                rf'{re.escape(symbol)}:\s*(\d+(?:\.\d+)?)%',
                rf'{re.escape(base_currency)}:\s*(\d+(?:\.\d+)?)%',
                rf'{re.escape(symbol.upper())}:\s*(\d+(?:\.\d+)?)%',
                rf'{re.escape(base_currency.upper())}:\s*(\d+(?:\.\d+)?)%',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    allocations[symbol] = float(match.group(1))
                    break
        
        # Validate allocations sum to 100% (with small tolerance)
        total_allocation = sum(allocations.values())
        
        # If allocations don't sum to 100% or we're missing some, use equal weights
        if abs(total_allocation - 100) > 5 or len(allocations) != len(recommended_cryptos):
            logger.warning(f"LLM allocations sum to {total_allocation}% or missing cryptos. Using equal weights.")
            equal_weight = 100.0 / len(recommended_cryptos)
            allocations = {symbol: equal_weight for symbol in recommended_cryptos}
        
        # Normalize to exactly 100%
        total_allocation = sum(allocations.values())
        if total_allocation > 0:
            allocations = {symbol: (pct / total_allocation) * 100 for symbol, pct in allocations.items()}
        
        decision = {
            "allocations": allocations,
            "reasoning": response,
            "total_capital": total_capital,
            "timestamp": crypto_data[0].get('timestamp') if crypto_data else None
        }
        
        # Format decision for human-readable logging
        logger.info("🤖 Portfolio Allocation:")
        for symbol, percentage in allocations.items():
            eur_amount = (percentage / 100) * total_capital
            logger.info(f"  {symbol}: {percentage:.1f}% (€{eur_amount:.2f})")
        logger.info(f"  Reasoning: {response}")
        
        return decision
