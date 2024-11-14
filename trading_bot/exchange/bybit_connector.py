"""
Bybit exchange connector with time synchronization
"""
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import asyncio
import platform
import time

# Set event loop policy for Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class BybitConnector:
    """Connector for Bybit exchange"""
    
    def __init__(self, api_key: str, api_secret: str, config: Dict):
        self.config = config
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': config.get('market_type', 'linear'),
                'defaultMarginType': config.get('margin_type', 'isolated'),
                'adjustForTimeDifference': True,
                'recvWindow': 60000  # Увеличиваем окно получения данных
            },
            'timeout': 30000,  # Увеличиваем таймаут
        })
        self.logger = logging.getLogger(__name__)
        self.time_offset = 0
        
    async def init(self):
        """Initialize connection and synchronize time"""
        try:
            await self._sync_time()
            self.logger.info("Time synchronized with Bybit server")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing connection: {str(e)}")
            return False
        
    async def _sync_time(self):
        """Synchronize local time with Bybit server time"""
        try:
            server_time = await self.exchange.fetch_time()
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            self.exchange.options['timeDifference'] = self.time_offset
            self.logger.info(f"Time offset: {self.time_offset}ms")
        except Exception as e:
            self.logger.error(f"Error synchronizing time: {str(e)}")
            raise
            
    async def fetch_historical_data(self, symbol: str, timeframe: str, 
                                  start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """Fetch historical data from Bybit"""
        try:
            # Ensure time is synchronized
            await self._sync_time()
            
            # Set end date to now if not provided
            end_date = end_date or datetime.now()
            
            # Calculate number of required candles
            timeframe_minutes = self._convert_timeframe_to_minutes(timeframe)
            total_minutes = int((end_date - start_date).total_seconds() / 60)
            required_candles = total_minutes // timeframe_minutes
            
            # Bybit limits
            limit = 1000  # Maximum candles per request
            all_candles = []
            
            # Fetch data in chunks
            current_start = start_date
            while current_start < end_date:
                try:
                    # Add timestamp adjustment
                    since = int(current_start.timestamp() * 1000) + self.time_offset
                    
                    candles = await self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since,
                        limit
                    )
                    
                    if not candles:
                        break
                        
                    all_candles.extend(candles)
                    
                    # Update start time for next request
                    last_candle_time = datetime.fromtimestamp(candles[-1][0] / 1000)
                    current_start = last_candle_time + timedelta(minutes=timeframe_minutes)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching candles: {str(e)}")
                    await asyncio.sleep(1)
                    continue
            
            if not all_candles:
                raise ValueError("No historical data received")
                
            # Convert to DataFrame
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Process DataFrame
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            
            # Remove duplicates and sort
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            # Filter by date range
            df = df[start_date:end_date]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            raise
    @staticmethod
    def _convert_timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        units = {
            'm': 1,
            'h': 60,
            'd': 1440,
            'w': 10080,
            'M': 43200
        }
        
        unit = timeframe[-1]
        if unit not in units:
            raise ValueError(f"Invalid timeframe unit: {unit}")
            
        try:
            number = int(timeframe[:-1])
            return number * units[unit]
        except ValueError:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

    async def setup_leverage(self, symbol: str, leverage: int) -> bool:
        """Setup leverage for symbol"""
        try:
            await self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            self.logger.error(f"Error setting leverage: {str(e)}")
            return False

    async def get_market_info(self, symbol: str) -> Dict:
        """Get market information"""
        try:
            market = await self.exchange.fetch_market(symbol)
            return {
                'symbol': market['symbol'],
                'base': market['base'],
                'quote': market['quote'],
                'price_precision': market['precision']['price'],
                'amount_precision': market['precision']['amount'],
                'minimum_amount': market['limits']['amount']['min'],
                'maximum_amount': market['limits']['amount']['max'],
                'minimum_cost': market['limits']['cost']['min'],
                'maximum_cost': market['limits']['cost']['max']
            }
        except Exception as e:
            self.logger.error(f"Error fetching market info: {str(e)}")
            raise

    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate"""
        try:
            funding = await self.exchange.fetch_funding_rate(symbol)
            return funding['fundingRate']
        except Exception as e:
            self.logger.error(f"Error fetching funding rate: {str(e)}")
            return 0.0

    async def close(self):
        """Close exchange connection"""
        try:
            await self.exchange.close()
        except Exception as e:
            self.logger.error(f"Error closing exchange connection: {str(e)}")

    async def __aenter__(self):
        """Context manager enter"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()
