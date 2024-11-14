"""
Script to run backtest
"""
import asyncio
import yaml
from datetime import datetime, timedelta
from trading_bot.exchange.bybit_connector import BybitConnector
from trading_bot.backtesting.engine import BacktestEngine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import platform
import os

# Setup event loop policy for Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        logger.info("Starting backtest process")
        
        # Load configuration
        logger.info("Loading configuration...")
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Initialize connector
        logger.info("Initializing Bybit connector...")
        connector = BybitConnector(
            config['api_key'],
            config['api_secret'],
            config
        )
        
        # Initialize connection
        if not await connector.init():
            raise Exception("Failed to initialize connection")
        
        try:
            # Fetch historical data
            logger.info("Fetching historical data...")
            start_date = datetime.now() - timedelta(days=config['backtest_days'])
            data = await connector.fetch_historical_data(
                config['symbol'],
                config['timeframe'],
                start_date
            )
            logger.info(f"Fetched {len(data)} candles")
            
            # Create results directory if it doesn't exist
            if not os.path.exists('results'):
                os.makedirs('results')
            
            # Save raw data
            data.to_csv('results/raw_data.csv')
            logger.info("Raw data saved to results/raw_data.csv")
            
            # Run backtest
            logger.info("Running backtest...")
            backtester = BacktestEngine(config)
            results = await backtester.run(data)
            
            # Print results
            logger.info("\nBacktest Results:")
            print(f"Total trades: {results.total_trades}")
            print(f"Winning trades: {results.winning_trades}")
            print(f"Losing trades: {results.losing_trades}")
            print(f"Win rate: {results.win_rate:.2%}")
            print(f"Profit factor: {results.profit_factor:.2f}")
            print(f"Total profit: ${results.total_profit:.2f}")
            print(f"Total fees: ${results.total_fees:.2f}")
            print(f"Max drawdown: {results.max_drawdown:.2%}")
            print(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
            
            # Plot results
            logger.info("Generating plots...")
            plt.figure(figsize=(15, 10))
            
            # Equity curve
            plt.subplot(2, 1, 1)
            plt.plot(results.equity_curve)
            plt.title('Equity Curve')
            plt.grid(True)
            
            # Drawdown
            plt.subplot(2, 1, 2)
            drawdown = (results.equity_curve.cummax() - results.equity_curve) / results.equity_curve.cummax()
            plt.plot(drawdown)
            plt.title('Drawdown')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('results/backtest_results.png')
            plt.close()
            
            # Save detailed results to CSV
            logger.info("Saving trade history...")
            trades_df = pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'side': t.side,
                    'size': t.size,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'status': t.status
                }
                for t in results.trades
            ])
            trades_df.to_csv('results/backtest_trades.csv', index=False)
            
            # Save metrics to file
            logger.info("Saving metrics...")
            metrics_df = pd.DataFrame([results.metrics])
            metrics_df.to_csv('results/backtest_metrics.csv', index=False)
            
            logger.info("Backtest completed successfully!")
            
        finally:
            logger.info("Closing connection...")
            await connector.close()
            
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
