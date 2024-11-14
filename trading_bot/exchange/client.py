"""
Bybit API client implementation
"""
import logging
import asyncio
from typing import Dict, List, Optional
import aiohttp
from datetime import datetime

class BybitClient:
    """Low-level client for Bybit API"""
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        self.session = None
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        """Context manager enter"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.disconnect()
        
    async def connect(self):
        """Create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    'Content-Type': 'application/json',
                    'api-key': self.api_key
                }
            )
            
    async def disconnect(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def get_server_time(self) -> int:
        """Get Bybit server time"""
        try:
            async with self.session.get(f"{self.base_url}/v3/public/time") as response:
                data = await response.json()
                return int(data['result']['timeNow'])
        except Exception as e:
            self.logger.error(f"Error getting server time: {e}")
            raise
            
    async def get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate"""
        try:
            params = {'symbol': symbol}
            async with self.session.get(
                f"{self.base_url}/v3/public/funding/prev-funding-rate",
                params=params
            ) as response:
                data = await response.json()
                return data['result']
        except Exception as e:
            self.logger.error(f"Error getting funding rate: {e}")
            raise
            
    async def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """Get orderbook data"""
        try:
            params = {'symbol': symbol, 'limit': limit}
            async with self.session.get(
                f"{self.base_url}/v3/public/orderbook/L2",
                params=params
            ) as response:
                data = await response.json()
                return data['result']
        except Exception as e:
            self.logger.error(f"Error getting orderbook: {e}")
            raise
            
    @staticmethod
    def generate_signature(params: Dict, secret: str) -> str:
        """Generate signature for authenticated requests"""
        # Implement signature generation according to Bybit docs
        # This is a placeholder - actual implementation needed
        return ""
