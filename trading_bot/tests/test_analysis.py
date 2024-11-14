"""
Tests for analysis module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_bot.analysis.indicators import TechnicalIndicators
from trading_bot.analysis.analyzer import MarketAnalyzer
from trading_bot.core.models import MarketData

@pytest.fixture
def sample_data():
    """Create sample market data"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')  # Changed 'H' to 'h'
    df = pd.DataFrame({
        'open': np.random.normal(50000, 1000, 100),
        'high': np.random.normal(51000, 1000, 100),
        'low': np.random.normal(49000, 1000, 100),
        'close': np.random.normal(50000, 1000, 100),
        'volume': np.random.normal(100, 10, 100)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df

@pytest.fixture
def empty_data():
    """Create empty DataFrame for testing edge cases"""
    return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

@pytest.fixture
def invalid_data():
    """Create invalid data for testing error handling"""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='h')
    return pd.DataFrame({
        'open': [np.nan, 50000, 50100, 50200, 50300],
        'high': [51000, np.nan, 51100, 51200, 51300],
        'low': [49000, 49100, np.nan, 49200, 49300],
        'close': [50000, 50100, 50200, np.nan, 50400],
        'volume': [100, 101, 102, 103, np.nan]
    }, index=dates)

# Existing tests remain the same...

def test_technical_indicators_empty_data(empty_data):
    """Test technical indicators with empty data"""
    # Act & Assert
    with pytest.raises(ValueError, match="Empty data provided"):
        TechnicalIndicators.calculate_all(empty_data)

def test_technical_indicators_invalid_data(invalid_data):
    """Test technical indicators with invalid data"""
    # Act
    result = TechnicalIndicators.calculate_all(invalid_data)
    
    # Assert
    assert result['ema_20'].isna().any()  # Should have some NaN values
    assert not result['ema_20'].isna().all()  # But not all should be NaN

def test_trend_detection_insufficient_data(sample_data):
    """Test trend detection with insufficient data"""
    # Arrange
    short_data = sample_data.iloc[-10:]  # Only 10 periods
    
    # Act
    trend = TechnicalIndicators.detect_trend(short_data)
    
    # Assert
    assert trend == "neutral"  # Should return neutral for insufficient data

@pytest.mark.asyncio
async def test_market_analyzer_cache(sample_data, config):
    """Test market analyzer caching mechanism"""
    # Arrange
    analyzer = MarketAnalyzer(config)
    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        open=sample_data['open'].iloc[-1],
        high=sample_data['high'].iloc[-1],
        low=sample_data['low'].iloc[-1],
        close=sample_data['close'].iloc[-1],
        volume=sample_data['volume'].iloc[-1]
    )
    
    # Act - First analysis
    analysis1 = await analyzer.analyze_market(market_data, sample_data)
    
    # Same request within cache timeout
    analysis2 = await analyzer.analyze_market(market_data, sample_data)
    
    # Assert
    assert analysis1 == analysis2
    assert len(analyzer.analysis_cache) == 1

@pytest.mark.asyncio
async def test_signal_generation_threshold(sample_data, config):
    """Test signal generation confidence threshold"""
    # Arrange
    analyzer = MarketAnalyzer({**config, 'min_signal_confidence': 0.9})  # High threshold
    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        open=sample_data['open'].iloc[-1],
        high=sample_data['high'].iloc[-1],
        low=sample_data['low'].iloc[-1],
        close=sample_data['close'].iloc[-1],
        volume=sample_data['volume'].iloc[-1]
    )
    
    # Act
    analysis = await analyzer.analyze_market(market_data, sample_data)
    signal = await analyzer.generate_signal(analysis, '1h')
    
    # Assert
    # With high threshold, should rarely generate signals
    if signal:
        assert signal.confidence >= 0.9

@pytest.mark.asyncio
async def test_market_analyzer_error_handling(empty_data, config):
    """Test market analyzer error handling"""
    # Arrange
    analyzer = MarketAnalyzer(config)
    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        open=0,
        high=0,
        low=0,
        close=0,
        volume=0
    )
    
    # Act & Assert
    with pytest.raises(ValueError):
        await analyzer.analyze_market(market_data, empty_data)

def test_support_resistance_min_data(sample_data):
    """Test support and resistance calculation with minimum data"""
    # Arrange
    min_data = sample_data.iloc[-5:]  # Only 5 periods
    
    # Act
    levels = TechnicalIndicators.calculate_support_resistance(min_data, window=3)
    
    # Assert
    assert levels['support'] <= levels['resistance']
    assert levels['mid_point'] == (levels['support'] + levels['resistance']) / 2
