"""
Tests for exchange module
"""
import pytest
from datetime import datetime
import ccxt.async_support as ccxt
from typing import Dict, List
from trading_bot.core.models import MarketData, Position
from trading_bot.exchange.repository import BybitRepository

@pytest.fixture
def mock_ohlcv_data() -> List:
    """Mock OHLCV data fixture"""
    timestamp = int(datetime.now().timestamp() * 1000)
    return [
        [timestamp, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],
        [timestamp + 60000, 50500.0, 51500.0, 50000.0, 51000.0, 150.0],
    ]

@pytest.fixture
def mock_position_data() -> Dict:
    """Mock position data fixture"""
    return {
        'symbol': 'BTCUSDT',
        'side': 'long',
        'contracts': '1.0',
        'entryPrice': '50000',
        'markPrice': '51000',
        'unrealizedPnl': '1000',
        'initialMargin': '5000',
        'liquidationPrice': '45000',
    }

@pytest.fixture
async def mock_exchange(mocker, mock_ohlcv_data, mock_position_data):
    """Mock ccxt.bybit exchange fixture"""
    exchange = mocker.Mock(spec=ccxt.bybit)
    
    # Mock fetch_ohlcv
    async def mock_fetch_ohlcv(*args, **kwargs):
        return mock_ohlcv_data
    exchange.fetch_ohlcv = mock_fetch_ohlcv
    
    # Mock fetch_positions
    async def mock_fetch_positions(*args, **kwargs):
        return [mock_position_data]
    exchange.fetch_positions = mock_fetch_positions
    
    # Mock create_order
    async def mock_create_order(*args, **kwargs):
        return {
            'id': '12345',
            'symbol': args[0],
            'type': args[1],
            'side': args[2],
            'amount': args[3],
            'price': args[4] if len(args) > 4 else None,
            'status': 'closed'
        }
    exchange.create_order = mock_create_order
    
    return exchange

@pytest.fixture
async def repository(mock_exchange):
    """Repository fixture with mocked exchange"""
    config = {
        'cache_timeout': 60,
        'testnet': True
    }
    repo = BybitRepository('test_key', 'test_secret', config)
    repo.exchange = mock_exchange
    return repo

@pytest.mark.asyncio
async def test_get_market_data(repository):
    """Test getting market data"""
    # Act
    market_data = await repository.get_market_data('BTCUSDT', '1m')
    
    # Assert
    assert isinstance(market_data, MarketData)
    assert market_data.symbol == 'BTCUSDT'
    assert market_data.open == 50000.0
    assert market_data.high == 51000.0
    assert market_data.low == 49000.0
    assert market_data.close == 50500.0
    assert market_data.volume == 100.0

@pytest.mark.asyncio
async def test_get_positions(repository):
    """Test getting positions"""
    # Act
    positions = await repository.get_positions()
    
    # Assert
    assert len(positions) == 1
    position = positions[0]
    assert isinstance(position, Position)
    assert position.symbol == 'BTCUSDT'
    assert position.side == 'long'
    assert position.size == 1.0
    assert float(position.entry_price) == 50000.0

@pytest.mark.asyncio
async def test_create_order(repository):
    """Test creating order"""
    # Arrange
    order_data = {
        'symbol': 'BTCUSDT',
        'type': 'limit',
        'side': 'buy',
        'amount': 1.0,
        'price': 50000.0
    }
    
    # Act
    result = await repository.create_order(order_data)
    
    # Assert
    assert result['symbol'] == 'BTCUSDT'
    assert result['type'] == 'limit'
    assert result['side'] == 'buy'
    assert result['amount'] == 1.0
    assert result['price'] == 50000.0
    assert result['status'] == 'closed'

@pytest.mark.asyncio
async def test_market_data_caching(repository):
    """Test market data caching mechanism"""
    # Arrange
    symbol = 'BTCUSDT'
    timeframe = '1m'
    
    # Act - First call (should hit the exchange)
    data1 = await repository.get_market_data(symbol, timeframe)
    # Second call (should hit the cache)
    data2 = await repository.get_market_data(symbol, timeframe)
    
    # Assert
    assert data1 == data2
    assert f"{symbol}_{timeframe}" in repository.cache

@pytest.mark.asyncio
async def test_error_handling_market_data(repository, mock_exchange):
    """Test error handling when fetching market data"""
    # Arrange
    async def mock_error(*args, **kwargs):
        raise ccxt.NetworkError("Network error")
    mock_exchange.fetch_ohlcv = mock_error
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await repository.get_market_data('BTCUSDT', '1m')
    assert "Network error" in str(exc_info.value)
