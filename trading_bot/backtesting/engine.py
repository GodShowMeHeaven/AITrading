"""
Backtesting engine for trading strategies
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from ..core.models import MarketData, Signal, Trade
from ..analysis.analyzer import MarketAnalyzer
from dataclasses import dataclass
import logging

@dataclass
class BacktestResult:
    """Results of backtest"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: float
    total_fees: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict
    parameters: Dict

class BacktestEngine:
    """Backtesting engine implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.analyzer = MarketAnalyzer(config)
        self.logger = logging.getLogger(__name__)
        self.commission = config.get('commission', 0.0006)  # Default Bybit fee
        self.initial_balance = config.get('initial_balance', 10000)
        self.leverage = config.get('leverage', 1)
        
    async def run(self, data: pd.DataFrame, params: Dict = None) -> BacktestResult:
        """Run backtest with given data and parameters"""
        try:
            # Initialize tracking variables
            balance = self.initial_balance
            position = 0
            trades = []
            equity_curve = []
            max_balance = balance
            max_drawdown = 0
            
            # Prepare data
            df = self._prepare_data(data)
            
            # Run simulation
            for i in range(len(df)):
                try:
                    # Get current window of data
                    current_data = df.iloc[:i+1]
                    if len(current_data) < self.config.get('min_data_points', 100):
                        continue
                        
                    # Create MarketData object for current candle
                    market_data = self._create_market_data(df.iloc[i])
                    
                    # Get analysis and signal
                    analysis = await self.analyzer.analyze_market(market_data, current_data)
                    signal = await self.analyzer.generate_signal(analysis, self.config['timeframe'])
                    
                    # Process signal
                    if signal:
                        # Close position if exists and signal is opposite
                        if position != 0 and (
                            (position > 0 and signal.action == 'sell') or
                            (position < 0 and signal.action == 'buy')
                        ):
                            trade = self._close_position(
                                position,
                                df.iloc[i],
                                balance,
                                trades[-1] if trades else None
                            )
                            trades.append(trade)
                            balance += trade.pnl - (abs(trade.size) * trade.exit_price * self.commission)
                            position = 0
                        
                        # Open new position if no position exists
                        elif position == 0:
                            size = self._calculate_position_size(balance, df.iloc[i]['close'])
                            if signal.action == 'buy':
                                position = size
                            elif signal.action == 'sell':
                                position = -size
                                
                            trades.append(Trade(
                                symbol=market_data.symbol,
                                side=signal.action,
                                size=abs(position),
                                entry_price=df.iloc[i]['close'],
                                exit_price=None,
                                pnl=None,
                                timestamp=market_data.timestamp,
                                status='open',
                                metadata={
                                    'signal_confidence': signal.confidence,
                                    'market_condition': analysis['market_condition']
                                }
                            ))
                    
                    # Track equity and drawdown
                    current_equity = self._calculate_equity(
                        balance,
                        position,
                        df.iloc[i]['close'],
                        trades[-1] if trades else None
                    )
                    equity_curve.append(current_equity)
                    
                    max_balance = max(max_balance, current_equity)
                    current_drawdown = (max_balance - current_equity) / max_balance
                    max_drawdown = max(max_drawdown, current_drawdown)
                    
                except Exception as e:
                    self.logger.error(f"Error processing timeframe {i}: {str(e)}")
                    continue
            
            # Close any remaining position
            if position != 0:
                trade = self._close_position(
                    position,
                    df.iloc[-1],
                    balance,
                    trades[-1]
                )
                trades.append(trade)
                balance += trade.pnl - (abs(trade.size) * trade.exit_price * self.commission)
            
            # Calculate final metrics
            metrics = self._calculate_metrics(trades, equity_curve, max_drawdown)
            
            return BacktestResult(
                total_trades=len(trades),
                winning_trades=len([t for t in trades if t.pnl and t.pnl > 0]),
                losing_trades=len([t for t in trades if t.pnl and t.pnl <= 0]),
                total_profit=balance - self.initial_balance,
                total_fees=sum(abs(t.size) * t.entry_price * self.commission for t in trades),
                max_drawdown=max_drawdown,
                sharpe_ratio=metrics['sharpe_ratio'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                trades=trades,
                equity_curve=pd.Series(equity_curve),
                metrics=metrics,
                parameters=params or {}
            )
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            raise
            
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting"""
        df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Remove rows with NaN values
        df = df.dropna()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        return df
        
    def _create_market_data(self, row: pd.Series) -> MarketData:
        """Create MarketData object from DataFrame row"""
        return MarketData(
            symbol=self.config['symbol'],
            timestamp=row.name if isinstance(row.name, datetime) else datetime.now(),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume'])
        )
        
    def _calculate_position_size(self, balance: float, price: float) -> float:
        """Calculate position size based on risk parameters"""
        risk_per_trade = self.config.get('risk_per_trade', 0.02)
        position_size = (balance * risk_per_trade * self.leverage) / price
        
        # Round to asset precision
        precision = self.config.get('asset_precision', 3)
        return round(position_size, precision)
        
    def _close_position(self, position: float, data: pd.Series, balance: float, 
                       entry_trade: Trade) -> Trade:
        """Close position and calculate PnL"""
        exit_price = data['close']
        pnl = position * (exit_price - entry_trade.entry_price)
        
        return Trade(
            symbol=entry_trade.symbol,
            side='sell' if position > 0 else 'buy',
            size=abs(position),
            entry_price=entry_trade.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            timestamp=data.name if isinstance(data.name, datetime) else datetime.now(),
            status='closed',
            metadata=entry_trade.metadata
        )
        
    def _calculate_equity(self, balance: float, position: float, 
                         current_price: float, open_trade: Optional[Trade]) -> float:
        """Calculate current equity including unrealized PnL"""
        if position == 0 or not open_trade:
            return balance
            
        unrealized_pnl = position * (current_price - open_trade.entry_price)
        return balance + unrealized_pnl
        
    def _calculate_metrics(self, trades: List[Trade], equity_curve: List[float], 
                          max_drawdown: float) -> Dict:
        """Calculate trading metrics"""
        closed_trades = [t for t in trades if t.status == 'closed' and t.pnl is not None]
        
        if not closed_trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'average_trade': 0
            }
            
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        total_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        
        # Calculate metrics
        win_rate = len(winning_trades) / len(closed_trades)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate Sharpe Ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 1 else 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'average_trade': sum(t.pnl for t in closed_trades) / len(closed_trades),
            'total_profit': total_profit,
            'total_loss': total_loss,
            'largest_win': max(t.pnl for t in closed_trades),
            'largest_loss': min(t.pnl for t in closed_trades),
            'average_win': total_profit / len(winning_trades) if winning_trades else 0,
            'average_loss': total_loss / len(losing_trades) if losing_trades else 0,
            'win_loss_ratio': (total_profit / len(winning_trades)) / (total_loss / len(losing_trades)) 
                            if winning_trades and losing_trades else 0
        }
