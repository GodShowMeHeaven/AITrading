"""
Advanced position size management with dynamic position sizing
"""
from typing import Dict, Optional, Tuple, List, Union
from datetime import datetime
import logging
import numpy as np

class DynamicPositionManager:
    """Manages position sizes based on various risk factors"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.initial_risk_per_trade = float(config.get('risk_per_trade', 0.02))
        self.max_position_size = float(config.get('max_position_size', 0.15))
        self.max_leverage = float(config.get('max_leverage', 3))
        self.equity_threshold = float(config.get('equity_threshold', 0.95))
        self.position_history: List[Dict] = []
        self.current_drawdown = 0.0
        self.peak_equity = 0.0

    def _convert_to_float(self, value: Union[float, np.floating, int]) -> float:
        """Convert numpy types to Python float"""
        return float(value)

    def calculate_position_size(
        self, 
        equity: float,
        price: float,
        volatility: float,
        signal_strength: float,
        market_conditions: Dict
    ) -> Tuple[float, Dict]:
        """Calculate dynamic position size"""
        try:
            # Update equity metrics
            self._update_equity_metrics(equity)
            
            # Base position size from equity
            base_size = self._convert_to_float(self._calculate_base_size(equity, price))
            
            # Risk adjustments
            volatility_factor = self._convert_to_float(self._volatility_adjustment(volatility))
            signal_factor = self._convert_to_float(self._signal_adjustment(signal_strength))
            market_factor = self._convert_to_float(self._market_condition_adjustment(market_conditions))
            drawdown_factor = self._convert_to_float(self._drawdown_adjustment())
            
            # Combined adjustment
            total_adjustment = (
                volatility_factor *
                signal_factor *
                market_factor *
                drawdown_factor
            )
            
            # Final position size
            position_size = self._convert_to_float(base_size * total_adjustment)
            
            # Apply limits
            position_size = self._convert_to_float(self._apply_limits(position_size, equity, price))
            
            # Risk metrics
            risk_metrics = {
                'position_size_usd': self._convert_to_float(position_size * price),
                'position_size_pct': self._convert_to_float((position_size * price) / equity),
                'risk_factors': {
                    'volatility': volatility_factor,
                    'signal': signal_factor,
                    'market': market_factor,
                    'drawdown': drawdown_factor
                },
                'risk_level': self._calculate_risk_level(position_size, equity, price),
                'leverage_used': self._convert_to_float((position_size * price) / equity),
                'current_drawdown': self._convert_to_float(self.current_drawdown)
            }
            
            return position_size, risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0, {}
            
    def _update_equity_metrics(self, current_equity: float) -> None:
        """Update equity peaks and drawdowns"""
        try:
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            if self.peak_equity > 0:
                self.current_drawdown = self._convert_to_float(
                    1 - (current_equity / self.peak_equity)
                )
                
        except Exception as e:
            self.logger.error(f"Error updating equity metrics: {str(e)}")
            
    def _calculate_base_size(self, equity: float, price: float) -> float:
        """Calculate base position size"""
        try:
            win_rate = self._calculate_win_rate()
            avg_win_loss_ratio = self._calculate_win_loss_ratio()
            
            if win_rate > 0 and avg_win_loss_ratio > 0:
                kelly_fraction = self._convert_to_float(
                    win_rate - ((1 - win_rate) / avg_win_loss_ratio)
                )
                kelly_fraction = self._convert_to_float(
                    max(0, min(kelly_fraction, self.initial_risk_per_trade))
                )
            else:
                kelly_fraction = self.initial_risk_per_trade
                
            return self._convert_to_float((equity * kelly_fraction) / price)
            
        except Exception as e:
            self.logger.error(f"Error calculating base size: {str(e)}")
            return 0.0

    def _volatility_adjustment(self, current_volatility: float) -> float:
        """Adjust position size based on volatility"""
        try:
            vol_percentile = self._calculate_volatility_percentile(current_volatility)
            
            if vol_percentile > 0.8:
                return 0.5
            elif vol_percentile > 0.6:
                return 0.75
            elif vol_percentile > 0.4:
                return 1.0
            elif vol_percentile > 0.2:
                return 1.25
            else:
                return 1.5
                
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 1.0
            
    def _signal_adjustment(self, signal_strength: float) -> float:
        """Adjust position size based on signal strength"""
        try:
            base_factor = 0.5
            return self._convert_to_float(base_factor + (signal_strength * (1 - base_factor)))
            
        except Exception as e:
            self.logger.error(f"Error calculating signal adjustment: {str(e)}")
            return 1.0
            
    def _market_condition_adjustment(self, conditions: Dict) -> float:
        """Adjust position size based on market conditions"""
        try:
            if not conditions:
                return 1.0
                
            trend_strength = float(conditions.get('trend_strength', 0))
            volatility_regime = conditions.get('volatility_regime', 'normal')
            liquidity = float(conditions.get('liquidity', 1.0))
            
            trend_factor = 1.0 + (0.5 * trend_strength)
            
            vol_factor = {
                'low': 1.2,
                'normal': 1.0,
                'high': 0.8
            }.get(volatility_regime, 1.0)
            
            liq_factor = min(1.0, liquidity)
            
            return self._convert_to_float(trend_factor * vol_factor * liq_factor)
            
        except Exception as e:
            self.logger.error(f"Error calculating market adjustment: {str(e)}")
            return 1.0
            
    def _drawdown_adjustment(self) -> float:
        """Adjust position size based on drawdown"""
        try:
            if self.current_drawdown > self.equity_threshold:
                reduction_factor = self._convert_to_float(
                    1 - ((self.current_drawdown - self.equity_threshold) / 
                         (1 - self.equity_threshold))
                )
                return max(0.25, reduction_factor)
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown adjustment: {str(e)}")
            return 1.0
            
    def _apply_limits(self, position_size: float, equity: float, price: float) -> float:
        """Apply position size limits"""
        try:
            max_size = (equity * self.max_position_size) / price
            position_size = min(position_size, max_size)
            
            max_leveraged_size = (equity * self.max_leverage) / price
            return self._convert_to_float(min(position_size, max_leveraged_size))
            
        except Exception as e:
            self.logger.error(f"Error applying position limits: {str(e)}")
            return 0.0
            
    def _calculate_risk_level(self, position_size: float, equity: float, price: float) -> str:
        """Calculate risk level"""
        try:
            position_value = position_size * price
            risk_percentage = self._convert_to_float(position_value / equity)
            
            if risk_percentage > 0.1:
                return 'high'
            elif risk_percentage > 0.05:
                return 'medium'
            return 'low'
                
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {str(e)}")
            return 'unknown'
            
    def _calculate_win_rate(self) -> float:
        """Calculate historical win rate"""
        try:
            if not self.position_history:
                return 0.5
                
            wins = sum(1 for trade in self.position_history if trade['pnl'] > 0)
            return self._convert_to_float(wins / len(self.position_history))
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {str(e)}")
            return 0.5
            
    def _calculate_win_loss_ratio(self) -> float:
        """Calculate win/loss ratio"""
        try:
            if not self.position_history:
                return 1.0
                
            wins = [t['pnl'] for t in self.position_history if t['pnl'] > 0]
            losses = [abs(t['pnl']) for t in self.position_history if t['pnl'] < 0]
            
            if not wins or not losses:
                return 1.0
                
            avg_win = self._convert_to_float(np.mean(wins))
            avg_loss = self._convert_to_float(np.mean(losses))
            
            return self._convert_to_float(avg_win / avg_loss if avg_loss > 0 else 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating win/loss ratio: {str(e)}")
            return 1.0
            
    def _calculate_volatility_percentile(self, current_volatility: float) -> float:
        """Calculate volatility percentile"""
        try:
            if not self.position_history:
                return 0.5
                
            historical_vols = [t.get('volatility', 0) for t in self.position_history]
            if not historical_vols:
                return 0.5
                
            return self._convert_to_float(
                sum(1 for v in historical_vols if v < current_volatility) / 
                len(historical_vols)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility percentile: {str(e)}")
            return 0.5
            
    def update_position_history(self, trade_data: Dict) -> None:
        """Update trade history"""
        try:
            self.position_history.append({
                'timestamp': datetime.now(),
                'size': float(trade_data.get('size', 0)),
                'pnl': float(trade_data.get('pnl', 0)),
                'volatility': float(trade_data.get('volatility', 0))
            })
            
            max_history = int(self.config.get('max_history_size', 1000))
            if len(self.position_history) > max_history:
                self.position_history = self.position_history[-max_history:]
                
        except Exception as e:
            self.logger.error(f"Error updating position history: {str(e)}")
