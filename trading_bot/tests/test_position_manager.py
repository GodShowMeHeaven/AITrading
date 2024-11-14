"""
Tests for dynamic position manager
"""
import unittest
from unittest.mock import Mock
import numpy as np
from trading_bot.risk.position_manager import DynamicPositionManager

class TestDynamicPositionManager(unittest.TestCase):
    def setUp(self):
        self.config = {
            'risk_per_trade': 0.02,
            'max_position_size': 0.15,
            'max_leverage': 3,
            'equity_threshold': 0.95
        }
        self.position_manager = DynamicPositionManager(self.config)

    def test_basic_position_calculation(self):
        """Test basic position size calculation"""
        # Arrange
        equity = 10000
        price = 100
        volatility = 0.02
        signal_strength = 0.8
        market_conditions = {
            'trend_strength': 0.6,
            'volatility_regime': 'normal',
            'liquidity': 1.0
        }
        
        # Act
        size, metrics = self.position_manager.calculate_position_size(
            equity, price, volatility, signal_strength, market_conditions
        )
        
        # Assert
        self.assertGreater(size, 0)
        self.assertEqual(metrics['position_size_usd'], size * price)
        self.assertEqual(metrics['position_size_pct'], (size * price) / equity)
        self.assertIn(metrics['risk_level'], ['low', 'medium', 'high'])

    def test_position_limits(self):
        """Test position size limits"""
        # Arrange
        equity = 10000
        price = 100
        volatility = 0.01
        signal_strength = 1.0
        market_conditions = {
            'trend_strength': 1.0,
            'volatility_regime': 'low',
            'liquidity': 1.0
        }
        
        # Act
        size, metrics = self.position_manager.calculate_position_size(
            equity, price, volatility, signal_strength, market_conditions
        )
        
        # Assert
        max_allowed_size = (equity * self.position_manager.max_position_size) / price
        self.assertLessEqual(size, max_allowed_size)
        self.assertLessEqual(metrics['leverage_used'], self.position_manager.max_leverage)

    def test_drawdown_adjustment(self):
        """Test position size reduction during drawdown"""
        # Arrange
        equity = 8000  # 20% drawdown from 10000
        self.position_manager.peak_equity = 10000
        price = 100
        volatility = 0.02
        signal_strength = 0.8
        market_conditions = {'trend_strength': 0.5}
        
        # Act
        size, metrics = self.position_manager.calculate_position_size(
            equity, price, volatility, signal_strength, market_conditions
        )
        
        # Get normal size for comparison
        self.position_manager.peak_equity = equity  # Reset drawdown
        normal_size, _ = self.position_manager.calculate_position_size(
            equity, price, volatility, signal_strength, market_conditions
        )
        
        # Assert
        self.assertLess(size, normal_size)
        self.assertAlmostEqual(metrics['current_drawdown'], 0.2, places=2)

    def test_market_conditions_impact(self):
        """Test impact of different market conditions"""
        # Arrange
        equity = 10000
        price = 100
        volatility = 0.02
        signal_strength = 0.8
        
        # Test different market conditions
        bullish_conditions = {
            'trend_strength': 0.9,
            'volatility_regime': 'normal',
            'liquidity': 1.0
        }
        
        bearish_conditions = {
            'trend_strength': 0.2,
            'volatility_regime': 'high',
            'liquidity': 0.8
        }
        
        # Act
        bullish_size, _ = self.position_manager.calculate_position_size(
            equity, price, volatility, signal_strength, bullish_conditions
        )
        
        bearish_size, _ = self.position_manager.calculate_position_size(
            equity, price, volatility, signal_strength, bearish_conditions
        )
        
        # Assert
        self.assertGreater(bullish_size, bearish_size)

    def test_update_position_history(self):
        """Test position history updates"""
        # Arrange
        trade_data = {
            'size': 1.0,
            'pnl': 100,
            'volatility': 0.02
        }
        
        # Act
        initial_length = len(self.position_manager.position_history)
        self.position_manager.update_position_history(trade_data)
        
        # Assert
        self.assertEqual(len(self.position_manager.position_history), initial_length + 1)
        latest_trade = self.position_manager.position_history[-1]
        self.assertEqual(latest_trade['size'], trade_data['size'])
        self.assertEqual(latest_trade['pnl'], trade_data['pnl'])
        self.assertEqual(latest_trade['volatility'], trade_data['volatility'])

    def test_max_history_limit(self):
        """Test position history size limit"""
        # Arrange
        self.position_manager.config['max_history_size'] = 2
        trade_data = {'size': 1.0, 'pnl': 100, 'volatility': 0.02}
        
        # Act
        for _ in range(3):  # Add 3 trades
            self.position_manager.update_position_history(trade_data)
        
        # Assert
        self.assertEqual(len(self.position_manager.position_history), 2)

if __name__ == '__main__':
    unittest.main()
