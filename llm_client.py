import json
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger
from config import config

class LLMClient:
    def __init__(self, base_url: str = None, model: str = None, api_key: str = None):
        self.base_url = base_url or config.llm_base_url
        self.model = model or config.llm_model
        self.api_key = api_key or config.llm_api_key
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate a response from the local LLM"""
        try:
            # Try Ollama format first (most common for local LLMs)
            response = await self._try_ollama_format(prompt, system_prompt)
            if response:
                return response
                
            # Try OpenAI-compatible format
            response = await self._try_openai_format(prompt, system_prompt)
            if response:
                return response
                
            # Try Anthropic format
            response = await self._try_anthropic_format(prompt, system_prompt)
            if response:
                return response
                
            raise Exception("Could not connect to any supported LLM format")
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "HOLD"  # Default to hold if LLM fails
    
    async def _try_ollama_format(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Try Ollama API format"""
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            async with self.session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "").strip()
            return None
        except Exception as e:
            logger.debug(f"Ollama format failed: {e}")
            return None
    
    async def _try_openai_format(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Try OpenAI-compatible API format"""
        try:
            url = f"{self.base_url}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            async with self.session.post(url, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"].strip()
            return None
        except Exception as e:
            logger.debug(f"OpenAI format failed: {e}")
            return None
    
    async def _try_anthropic_format(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Try Anthropic API format"""
        try:
            url = f"{self.base_url}/v1/messages"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["x-api-key"] = self.api_key
                
            payload = {
                "model": self.model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            async with self.session.post(url, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["content"][0]["text"].strip()
            return None
        except Exception as e:
            logger.debug(f"Anthropic format failed: {e}")
            return None
    
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
        
        prompt = f"""Market Data Analysis:
        
        Symbol: {market_data.get('symbol', 'Unknown')}
        Current Price: {market_data.get('current_price', 'Unknown')}
        Price Change 24h: {market_data.get('price_change_24h', 'Unknown')}%
        Volume 24h: {market_data.get('volume_24h', 'Unknown')}
        RSI: {market_data.get('rsi', 'Unknown')}
        MACD: {market_data.get('macd', 'Unknown')}
        Moving Average (20): {market_data.get('ma_20', 'Unknown')}
        Moving Average (50): {market_data.get('ma_50', 'Unknown')}
        
        Current Position: {market_data.get('current_position', 'None')}
        Portfolio Value: {market_data.get('portfolio_value', 'Unknown')}
        
        Based on this data, what should I do? Respond with only the action and brief reason."""
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Parse the response
        action = "HOLD"
        reason = ""
        
        if response:
            response_upper = response.upper()
            if "BUY" in response_upper:
                action = "BUY"
            elif "SELL" in response_upper:
                action = "SELL"
            elif "CLOSE" in response_upper:
                action = "CLOSE"
            else:
                action = "HOLD"
            
            reason = response.replace(action, "").strip()
        
        return {
            "action": action,
            "reason": reason,
            "confidence": 0.8,  # Placeholder - could be enhanced with LLM confidence scoring
            "timestamp": market_data.get("timestamp")
        } 