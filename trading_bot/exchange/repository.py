"""
Bybit exchange repository implementation
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
import ccxt.async_support as ccxt
from ..core.interfaces import IExchangeRepository
from ..core.models import MarketData, Position

class BybitRepository(IExchangeRepository):
    """Implementation of exchange repository for Bybit"""
    def __init__(self, api_key: str, api_secret: str, config: Dict):
        self.config = config
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
            }
        })
        self.cache = {}
        self.cache_timeout = config.get('cache_timeout', 60)
        self.logger = logging.getLogger(__name__)
        
    async def get_market_data(self, symbol: str, timeframe: str) -> MarketData:
        """Get market data with caching"""
        cache_key = f"{symbol}_{timeframe}"
        
        try:
            # Check cache
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_timeout:
                    return cached_data['data']
            
            # Fetch new data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe)
            if not ohlcv:
                raise ValueError(f"No data received for {symbol}")
                
            # Convert and cache
            data = self._convert_to_market_data(ohlcv[-1], symbol)
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            raise
            
    async def create_order(self, order: Dict) -> Dict:
        """Create new order"""
        try:
            result = await self.exchange.create_order(
                symbol=order['symbol'],
                type=order['type'],
                side=order['side'],
                amount=order['amount'],
                price=order.get('price'),
                params=order.get('params', {})
            )
            return result
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            raise
            
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        try:
            raw_positions = await self.exchange.fetch_positions()
            return [self._convert_to_position(p) for p in raw_positions if float(p['contracts']) != 0]
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            raise
            
    def _convert_to_market_data(self, raw_data: List, symbol: str) -> MarketData:
        """Convert raw OHLCV data to MarketData model"""
        return MarketData(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(raw_data[0] / 1000),
            open=float(raw_data[1]),
            high=float(raw_data[2]),
            low=float(raw_data[3]),
            close=float(raw_data[4]),
            volume=float(raw_data[5])
        )
        
    def _convert_to_position(self, raw_position: Dict) -> Position:
        """Convert raw position data to Position model"""
        return Position(
            symbol=raw_position['symbol'],
            side=raw_position['side'],
            size=float(raw_position['contracts']),
            entry_price=float(raw_position['entryPrice']),
            current_price=float(raw_position['markPrice']),
            pnl=float(raw_position['unrealizedPnl']),
            margin=float(raw_position['initialMargin']),
            liquidation_price=float(raw_position['liquidationPrice']),
            timestamp=datetime.now()
        )
