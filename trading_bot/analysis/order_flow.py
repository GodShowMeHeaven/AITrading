"""
Order Flow Analysis implementation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

@dataclass
class OrderFlowData:
    """Order flow analysis data"""
    imbalance_ratio: float
    absorption_ratio: float
    delta: float
    cumulative_delta: float
    volume_profile: Dict[float, float]
    support_levels: List[float]
    resistance_levels: List[float]
    liquidity_levels: Dict[float, float]

class OrderFlowAnalyzer:
    """Advanced order flow analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tick_size = config.get('tick_size', 0.5)
        self.volume_profile_levels = config.get('volume_profile_levels', 50)
        
    def analyze_order_flow(self, trades: pd.DataFrame, orderbook: pd.DataFrame) -> OrderFlowData:
        """Analyze order flow data"""
        try:
            # Calculate buy/sell imbalance
            imbalance = self._calculate_imbalance(trades)
            
            # Calculate volume absorption
            absorption = self._calculate_absorption(trades, orderbook)
            
            # Calculate delta and cumulative delta
            delta, cum_delta = self._calculate_delta(trades)
            
            # Generate volume profile
            volume_profile = self._generate_volume_profile(trades)
            
            # Find support/resistance levels
            support, resistance = self._find_sr_levels(trades, volume_profile)
            
            # Analyze liquidity levels
            liquidity = self._analyze_liquidity(orderbook)
            
            return OrderFlowData(
                imbalance_ratio=imbalance,
                absorption_ratio=absorption,
                delta=delta,
                cumulative_delta=cum_delta,
                volume_profile=volume_profile,
                support_levels=support,
                resistance_levels=resistance,
                liquidity_levels=liquidity
            )
            
        except Exception as e:
            self.logger.error(f"Error in order flow analysis: {str(e)}")
            raise
            
    def _calculate_imbalance(self, trades: pd.DataFrame) -> float:
        """Calculate buy/sell imbalance"""
        try:
            buy_volume = trades[trades['side'] == 'buy']['volume'].sum()
            sell_volume = trades[trades['side'] == 'sell']['volume'].sum()
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0
                
            return (buy_volume - sell_volume) / total_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating imbalance: {str(e)}")
            return 0
            
    def _calculate_absorption(self, trades: pd.DataFrame, orderbook: pd.DataFrame) -> float:
        """Calculate volume absorption ratio"""
        try:
            # Calculate absorbed volume
            absorbed_volume = trades['volume'].where(
                (trades['price'] >= orderbook['asks'].min()) |
                (trades['price'] <= orderbook['bids'].max())
            ).sum()
            
            total_volume = trades['volume'].sum()
            
            if total_volume == 0:
                return 0
                
            return absorbed_volume / total_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating absorption: {str(e)}")
            return 0
            
    def _calculate_delta(self, trades: pd.DataFrame) -> Tuple[float, float]:
        """Calculate delta and cumulative delta"""
        try:
            trades['delta'] = trades.apply(
                lambda x: x['volume'] if x['side'] == 'buy' else -x['volume'],
                axis=1
            )
            
            current_delta = trades['delta'].sum()
            cumulative_delta = trades['delta'].cumsum()
            
            return current_delta, cumulative_delta.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating delta: {str(e)}")
            return 0.0, 0.0
            
    def _generate_volume_profile(self, trades: pd.DataFrame) -> Dict[float, float]:
        """Generate volume profile"""
        try:
            # Calculate price levels
            price_min = trades['price'].min()
            price_max = trades['price'].max()
            level_height = (price_max - price_min) / self.volume_profile_levels
            
            # Initialize levels
            levels = {
                round(price_min + i * level_height, 2): 0 
                for i in range(self.volume_profile_levels + 1)
            }
            
            # Distribute volume
            for _, trade in trades.iterrows():
                level = round(
                    price_min + 
                    int((trade['price'] - price_min) / level_height) * level_height,
                    2
                )
                levels[level] += trade['volume']
                
            return levels
            
        except Exception as e:
            self.logger.error(f"Error generating volume profile: {str(e)}")
            return {}
            
    def _find_sr_levels(self, trades: pd.DataFrame, 
                       volume_profile: Dict[float, float]) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        try:
            # Sort levels by volume
            sorted_levels = sorted(
                volume_profile.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Find high volume nodes
            high_volume_levels = [
                level[0] for level in sorted_levels[:5]
            ]
            
            # Separate into support and resistance
            current_price = trades['price'].iloc[-1]
            support = sorted([l for l in high_volume_levels if l < current_price])
            resistance = sorted([l for l in high_volume_levels if l > current_price])
            
            return support, resistance
            
        except Exception as e:
            self.logger.error(f"Error finding S/R levels: {str(e)}")
            return [], []
            
    def _analyze_liquidity(self, orderbook: pd.DataFrame) -> Dict[float, float]:
        """Analyze liquidity levels"""
        try:
            liquidity = {}
            
            # Analyze bid side
            for price, volume in orderbook['bids'].items():
                liquidity[price] = volume
                
            # Analyze ask side
            for price, volume in orderbook['asks'].items():
                liquidity[price] = -volume  # Negative for ask side
                
            return liquidity
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity: {str(e)}")
            return {}

    def get_trading_signals(self, of_data: OrderFlowData) -> Dict:
        """Generate trading signals from order flow analysis"""
        try:
            signals = {
                'strength': 0,
                'direction': 'neutral',
                'confidence': 0.0,
                'entry_zones': [],
                'exit_zones': []
            }
            
            # Calculate signal strength based on multiple factors
            imbalance_score = abs(of_data.imbalance_ratio)
            absorption_score = of_data.absorption_ratio
            delta_score = abs(of_data.delta) / (abs(of_data.cumulative_delta) + 1e-6)
            
            # Combine scores
            total_score = (
                imbalance_score * 0.4 +
                absorption_score * 0.3 +
                delta_score * 0.3
            )
            
            signals['strength'] = total_score
            
            # Determine direction
            if of_data.imbalance_ratio > 0.2 and of_data.delta > 0:
                signals['direction'] = 'long'
            elif of_data.imbalance_ratio < -0.2 and of_data.delta < 0:
                signals['direction'] = 'short'
                
            # Calculate confidence
            signals['confidence'] = min(0.95, total_score)
            
            # Identify entry and exit zones
            if signals['direction'] == 'long':
                signals['entry_zones'] = of_data.support_levels
                signals['exit_zones'] = of_data.resistance_levels
            else:
                signals['entry_zones'] = of_data.resistance_levels
                signals['exit_zones'] = of_data.support_levels
                
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            return {
                'strength': 0,
                'direction': 'neutral',
                'confidence': 0.0,
                'entry_zones': [],
                'exit_zones': []
            }
