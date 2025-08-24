import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    """Backtesting engine for trading strategies with capital management"""
    
    def __init__(self, strategy):
        self.strategy = strategy
        
    def run_backtest(self, historical_data):
        """Run backtest on historical data using strategy's capital management"""
        try:
            if historical_data is None or len(historical_data) == 0:
                return None
            
            # Use the strategy's own backtest implementation which includes capital management
            df, trades, summary, capital_dates, capital_over_time = self.strategy.prepare_data_and_backtest(historical_data)
            
            # Convert trades to DataFrame for compatibility
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            
            # Calculate cumulative P&L if we have trades
            if not trades_df.empty:
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            # Store additional data for plotting
            self.summary = summary
            self.capital_dates = capital_dates
            self.capital_over_time = capital_over_time
            
            return df, trades_df
            
        except Exception as e:
            print(f"Error in backtesting: {str(e)}")
            return None
    
    def get_summary(self):
        """Get backtest summary if available"""
        return getattr(self, 'summary', {})
    
    def get_capital_data(self):
        """Get capital over time data if available"""
        capital_dates = getattr(self, 'capital_dates', [])
        capital_over_time = getattr(self, 'capital_over_time', [])
        return capital_dates, capital_over_time
    
    def get_performance_metrics(self, trades_df):
        """Calculate performance metrics"""
        if trades_df.empty:
            return {}
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate maximum drawdown
        if 'cumulative_pnl' in trades_df.columns:
            peak = trades_df['cumulative_pnl'].expanding().max()
            drawdown = trades_df['cumulative_pnl'] - peak
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }
