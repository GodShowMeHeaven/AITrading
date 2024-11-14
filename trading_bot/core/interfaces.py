"""
Core interfaces for the trading bot
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from .models import MarketData, Position

class IExchangeRepository(ABC):
    """Interface for exchange operations"""
    @abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str) -> MarketData:
        """Get market data from exchange"""
        pass
        
    @abstractmethod
    async def create_order(self, order: Dict) -> Dict:
        """Create new order"""
        pass
        
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
