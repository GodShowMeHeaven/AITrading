"""
Technical indicators implementation
"""
import pandas as pd
import numpy as np
from typing import Dict
import ta
import logging

class TechnicalIndicators:
    """Technical analysis indicators calculator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            # Validate input
            if df.empty:
                raise ValueError("Empty data provided")
            
            # Create copy and handle missing data
            df = df.copy()
            df = df.ffill()  # Forward fill
            df = df.bfill()  # Back fill
            
            # Trend Indicators
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
            
            # ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['di_plus'] = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()
            
            # Volatility Indicators
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bb.bollinger_hband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            
            # Momentum Indicators
            df['rsi'] = ta.momentum.rsi(df['close'])
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def detect_trend(self, df: pd.DataFrame) -> str:
        """Detect current market trend"""
        try:
            if len(df) < 20:
                return "neutral"
            
            current = df.iloc[-1]
            
            # Calculate trend strength
            adx = df['adx'].iloc[-1]
            trend_strength = "strong" if adx > 25 else "weak"
            
            if current['ema_20'] > current['ema_50'] > current['ema_200']:
                return f"{trend_strength}_uptrend"
            elif current['ema_20'] < current['ema_50'] < current['ema_200']:
                return f"{trend_strength}_downtrend"
            else:
                return "neutral"
                
        except Exception as e:
            self.logger.error(f"Error detecting trend: {str(e)}")
            return "neutral"

    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate support and resistance levels"""
        try:
            if len(df) < window:
                window = len(df)
            
            highs = df['high'].rolling(window=window).max()
            lows = df['low'].rolling(window=window).min()
            
            current_high = highs.iloc[-1]
            current_low = lows.iloc[-1]
            
            return {
                'support': current_low,
                'resistance': current_high,
                'mid_point': (current_high + current_low) / 2
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {str(e)}")
            raise
