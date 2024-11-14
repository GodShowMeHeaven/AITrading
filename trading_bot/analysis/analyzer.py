"""
Market analyzer implementation
"""
import logging
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from .indicators import TechnicalIndicators
from ..core.models import MarketData, Signal

class MarketAnalyzer:
    """Market analysis and signal generation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators = TechnicalIndicators()
        self.logger = logging.getLogger(__name__)
        self.analysis_cache = {}
        self.cache_timeout = config.get('analysis_cache_timeout', 60)
        
    def _extract_technical_indicators(self, current: pd.Series) -> Dict:
        """Extract technical indicators from current data"""
        return {
            'adx': float(current.get('adx', 0)),
            'rsi': float(current.get('rsi', 50)),
            'macd': float(current.get('macd', 0)),
            'macd_signal': float(current.get('macd_signal', 0)),
            'bb_width': float(current.get('bb_width', 0))
        }

    def _analyze_price_action(self, current: pd.Series, df: pd.DataFrame) -> Dict:
        """Analyze price action patterns"""
        return {
            'above_ema_20': bool(current['close'] > current.get('ema_20', 0)),
            'above_ema_50': bool(current['close'] > current.get('ema_50', 0)),
            'above_ema_200': bool(current['close'] > current.get('ema_200', 0)),
            'volatility': float(current.get('bb_width', 0))
        }

    def _analyze_volatility(self, df: pd.DataFrame) -> str:
        """Analyze volatility state"""
        try:
            bb_width = df['bb_width'].iloc[-20:].mean()
            if bb_width > 2.0:
                return 'high'
            elif bb_width > 1.0:
                return 'medium'
            else:
                return 'low'
        except Exception:
            return 'unknown'

    def _determine_market_condition(self, analysis: Dict) -> str:
        """Determine market condition"""
        try:
            indicators = analysis['technical_indicators']
            volatility = analysis['price_action']['volatility']
            
            conditions = []
            
            # Trend strength
            if indicators['adx'] > 25:
                if analysis['trend'].startswith('strong_up'):
                    conditions.append('trending_bullish')
                elif analysis['trend'].startswith('strong_down'):
                    conditions.append('trending_bearish')
            
            # Volatility
            if volatility > 2.0:
                conditions.append('volatile')
            
            # Range conditions
            if 40 < indicators['rsi'] < 60 and volatility < 1.0:
                conditions.append('ranging')
            
            return conditions[0] if conditions else 'mixed'
            
        except Exception as e:
            self.logger.error(f"Error determining market condition: {str(e)}")
            return 'unknown'

    async def analyze_market(self, data: MarketData, historical_data: pd.DataFrame) -> Dict:
        """Analyze market conditions"""
        try:
            # Prepare data
            df = historical_data.copy()
            
            # Calculate indicators
            df = self.indicators.calculate_all(df)
            
            # Get current values
            current = df.iloc[-1]
            
            # Calculate components
            trend = self.indicators.detect_trend(df)
            support_resistance = self.indicators.calculate_support_resistance(df)
            
            analysis = {
                'symbol': data.symbol,
                'timestamp': data.timestamp,
                'trend': trend,
                'support_resistance': support_resistance,
                'technical_indicators': self._extract_technical_indicators(current),
                'price_action': self._analyze_price_action(current, df),
                'volatility_state': self._analyze_volatility(df)
            }
            
            # Add market condition
            analysis['market_condition'] = self._determine_market_condition(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Critical error in market analysis: {str(e)}")
            raise

    async def generate_signal(self, analysis: Dict, timeframe: str) -> Optional[Signal]:
        """Generate trading signal"""
        try:
            # Validate analysis data
            if not analysis or 'technical_indicators' not in analysis:
                return None
            
            indicators = analysis['technical_indicators']
            rsi = indicators['rsi']
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            adx = indicators['adx']
            
            # Initialize signal parameters
            action = 'hold'
            confidence = 0.0
            
            # Generate signal based on market conditions
            if analysis['market_condition'] == 'trending_bullish':
                if rsi < 70 and macd > macd_signal:
                    action = 'buy'
                    confidence = min(0.8 + (adx - 25) / 100, 0.95)
            
            elif analysis['market_condition'] == 'trending_bearish':
                if rsi > 30 and macd < macd_signal:
                    action = 'sell'
                    confidence = min(0.8 + (adx - 25) / 100, 0.95)
            
            elif analysis['market_condition'] == 'ranging':
                if rsi < 30:
                    action = 'buy'
                    confidence = 0.6
                elif rsi > 70:
                    action = 'sell'
                    confidence = 0.6
            
            # Return signal if confidence meets threshold
            if confidence >= self.config.get('min_signal_confidence', 0.6):
                return Signal(
                    symbol=analysis.get('symbol', 'unknown'),
                    action=action,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    params={
                        'timeframe': timeframe,
                        'market_condition': analysis['market_condition'],
                        'trend': analysis['trend']
                    },
                    metadata=analysis
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return None
