import pandas as pd
import numpy as np

class TradingStrategy:
    """21 EMA Trading Strategy Implementation with Capital Management"""
    
    def __init__(self, ema_period=21, starting_capital=100, leverage=50, 
                 risk_pct=0.02, target_pct=0.05, stoploss_points=500,
                 maker_fee_pct=0.0002, taker_fee_pct=0.0005):
        self.ema_period = ema_period
        self.starting_capital = starting_capital
        self.leverage = leverage
        self.risk_pct = risk_pct
        self.target_pct = target_pct
        self.stoploss_points = stoploss_points
        self.maker_fee_pct = maker_fee_pct  # 0.02% for maker orders
        self.taker_fee_pct = taker_fee_pct  # 0.05% for taker orders
        
    def calculate_ema(self, df, period=21):
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def prepare_data_and_backtest(self, df):
        """Prepare data and run complete backtest with capital management"""
        # Calculate EMA and prepare signals using your working logic
        df['EMA21'] = self.calculate_ema(df, self.ema_period)
        df['position'] = 0
        df['candle_color'] = np.where(df['close'] > df['open'], 'green', 'red')
        
        # Generate position signals exactly like your working code
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['EMA21'].iloc[i] and df['candle_color'].iloc[i] == 'green':
                df.loc[df.index[i], 'position'] = 1
            elif df['close'].iloc[i] < df['EMA21'].iloc[i] and df['candle_color'].iloc[i] == 'red':
                df.loc[df.index[i], 'position'] = -1
            else:
                df.loc[df.index[i], 'position'] = 0
        
        # Forward fill positions and calculate trade signals
        df['position'] = df['position'].replace(0, pd.NA).ffill().fillna(0)
        df['trade_signal'] = df['position'].diff()
        
        # Run the capital management backtest
        return self._run_capital_backtest(df)
    
    def _run_capital_backtest(self, df):
        """Run backtest with proper capital and risk management"""
        capital = self.starting_capital
        capital_over_time = []
        capital_dates = []
        trades = []
        open_trade = None
        
        for i in range(1, len(df)):
            if capital <= 0:
                print("Capital has depleted to zero or below. Stopping trading.")
                break
            
            signal_change = df['trade_signal'].iloc[i] != 0
            capital_dates.append(df.index[i])
            capital_over_time.append(capital)
            
            # Handle open trade exits
            if open_trade is not None:
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                entry = open_trade['entry_price']
                pos = open_trade['position']
                
                # Calculate position sizing based on capital and risk
                risk_dollars = capital * self.risk_pct
                target_dollars = capital * self.target_pct
                
                max_trade_value = capital * self.leverage
                units = max_trade_value / entry
                
                # Adjust units based on risk management
                sl_price_dist = self.stoploss_points
                max_units_by_risk = risk_dollars / sl_price_dist
                units = min(units, max_units_by_risk)
                
                # Calculate stop loss and target levels
                sl_level = entry - sl_price_dist if pos == 1 else entry + sl_price_dist
                target_price_dist = target_dollars / units
                target_level = entry + target_price_dist if pos == 1 else entry - target_price_dist
                
                # Calculate capital used and position value for this trade
                position_value = units * entry
                capital_used = position_value / self.leverage
                
                # Calculate liquidation price using Delta Exchange official formula
                # At liquidation: Position Margin - Unrealized PnL = Maintenance Margin
                
                # Calculate position size in BTC for maintenance margin determination
                position_size_btc = units * entry / 1000000  # Convert to BTC (assuming BTCUSD)
                
                # Delta Exchange BTCUSD Maintenance Margin tiers
                if position_size_btc <= 5:
                    mm_percent = 0.005  # 0.5% for positions <= 5 BTC
                else:
                    mm_percent = 0.005 + 0.00075 * (position_size_btc - 5)  # 0.5% + 0.075% * (size - 5)
                
                # Position margin (capital used for this position)
                position_margin = capital_used
                
                # Maintenance margin requirement
                maintenance_margin = position_value * mm_percent
                
                # Calculate liquidation price
                # For long: entry - (position_margin - maintenance_margin) / units
                # For short: entry + (position_margin - maintenance_margin) / units
                liquidation_distance = (position_margin - maintenance_margin) / units
                liquidation_level = entry - liquidation_distance if pos == 1 else entry + liquidation_distance
                
                exit_reason = None
                exit_price = None
                
                # Check exit conditions (including liquidation)
                if pos == 1:  # Long position
                    if low <= liquidation_level:
                        exit_price = liquidation_level
                        exit_reason = 'Liquidation'
                    elif low <= sl_level:
                        exit_price = sl_level
                        exit_reason = 'Stoploss'
                    elif high >= target_level:
                        exit_price = target_level
                        exit_reason = 'Target'
                else:  # Short position
                    if high >= liquidation_level:
                        exit_price = liquidation_level
                        exit_reason = 'Liquidation'
                    elif high >= sl_level:
                        exit_price = sl_level
                        exit_reason = 'Stoploss'
                    elif low <= target_level:
                        exit_price = target_level
                        exit_reason = 'Target'
                
                # Check for signal change exit
                if exit_reason is None and signal_change:
                    exit_price = df['open'].iloc[i]
                    exit_reason = 'Signal Change'
                
                # Process trade exit
                if exit_price is not None:
                    pnl_points = (exit_price - entry) if pos == 1 else (entry - exit_price)
                    pnl_before_fees = pnl_points * units
                    
                    # Calculate trading fees (entry + exit)
                    position_value = units * entry
                    entry_fee = position_value * self.taker_fee_pct  # Assume taker for entry
                    exit_fee = units * exit_price * self.taker_fee_pct  # Assume taker for exit
                    total_fees = entry_fee + exit_fee
                    
                    # Subtract fees from P&L
                    pnl_capital = pnl_before_fees - total_fees
                    capital += pnl_capital
                    
                    # Capital used and position value already calculated above
                    
                    # Handle liquidation impact on capital
                    if exit_reason == 'Liquidation':
                        # In liquidation, typically lose most of the margin
                        liquidation_loss = capital_used * 0.8  # Lose 80% of margin
                        capital -= liquidation_loss
                        pnl_capital = -liquidation_loss  # Override PnL for liquidation
                        total_fees = entry_fee  # Only entry fee charged in liquidation
                        exit_fee = 0
                    
                    trades.append({
                        'entry_time': open_trade['entry_time'],
                        'exit_time': df.index[i],
                        'side': 'long' if pos == 1 else 'short',
                        'entry_price': entry,
                        'exit_price': exit_price,
                        'pnl': pnl_capital,
                        'pnl_before_fees': pnl_before_fees,
                        'pnl_points': pnl_points,
                        'trading_fees': total_fees,
                        'entry_fee': entry_fee,
                        'exit_fee': exit_fee,
                        'exit_reason': exit_reason,
                        'stoploss': sl_level,
                        'target': target_level,
                        'liquidation_price': liquidation_level,
                        'units': units,
                        'risk_dollars': risk_dollars,
                        'target_dollars': target_dollars,
                        'position_value': position_value,
                        'capital_used': capital_used
                    })
                    open_trade = None
            
            # Handle new trade entries
            if signal_change and df['position'].iloc[i] != 0 and open_trade is None:
                open_trade = {
                    'entry_time': df.index[i],
                    'entry_price': df['open'].iloc[i],
                    'position': df['position'].iloc[i]
                }
        
        # Close any remaining open trade
        if open_trade is not None and capital > 0 and len(df) > 0:
            last_index = len(df) - 1
            close_price = df['close'].iloc[last_index]
            pos = open_trade['position']
            
            risk_dollars = capital * self.risk_pct
            target_dollars = capital * self.target_pct
            max_trade_value = capital * self.leverage
            units = max_trade_value / open_trade['entry_price']
            sl_price_dist = self.stoploss_points
            max_units_by_risk = risk_dollars / sl_price_dist
            units = min(units, max_units_by_risk)
            
            pnl_points = (close_price - open_trade['entry_price']) if pos == 1 else (open_trade['entry_price'] - close_price)
            pnl_before_fees = pnl_points * units
            
            # Calculate trading fees for final position
            position_value = units * open_trade['entry_price']
            entry_fee = position_value * self.taker_fee_pct
            exit_fee = units * close_price * self.taker_fee_pct
            total_fees = entry_fee + exit_fee
            
            pnl_capital = pnl_before_fees - total_fees
            capital += pnl_capital
            
            # Calculate capital used for this trade
            position_value = units * open_trade['entry_price']
            capital_used = position_value / self.leverage
            
            # Calculate liquidation price for the final trade using Delta Exchange formula
            position_size_btc = units * open_trade['entry_price'] / 1000000  # Convert to BTC
            
            # Delta Exchange BTCUSD Maintenance Margin tiers
            if position_size_btc <= 5:
                mm_percent = 0.005  # 0.5% for positions <= 5 BTC
            else:
                mm_percent = 0.005 + 0.00075 * (position_size_btc - 5)  # 0.5% + 0.075% * (size - 5)
            
            maintenance_margin = position_value * mm_percent
            liquidation_distance = (capital_used - maintenance_margin) / units
            liquidation_level = open_trade['entry_price'] - liquidation_distance if pos == 1 else open_trade['entry_price'] + liquidation_distance
            
            trades.append({
                'entry_time': open_trade['entry_time'],
                'exit_time': df.index[last_index],
                'side': 'long' if pos == 1 else 'short',
                'entry_price': open_trade['entry_price'],
                'exit_price': close_price,
                'pnl': pnl_capital,
                'pnl_before_fees': pnl_before_fees,
                'pnl_points': pnl_points,
                'trading_fees': total_fees,
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'exit_reason': 'EOD Close',
                'stoploss': open_trade['entry_price'] - sl_price_dist if pos == 1 else open_trade['entry_price'] + sl_price_dist,
                'target': open_trade['entry_price'] + target_dollars/units if pos == 1 else open_trade['entry_price'] - target_dollars/units,
                'liquidation_price': liquidation_level,
                'units': units,
                'risk_dollars': risk_dollars,
                'target_dollars': target_dollars,
                'position_value': position_value,
                'capital_used': capital_used
            })
            capital_dates.append(df.index[last_index])
            capital_over_time.append(capital)
        
        # Calculate summary statistics
        total_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = total_trades - wins
        net_profit = capital - self.starting_capital
        accuracy = wins / total_trades if total_trades > 0 else 0
        
        summary = {
            'starting_capital': self.starting_capital,
            'ending_capital': capital,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'accuracy': accuracy,
            'net_profit': net_profit
        }
        
        return df, trades, summary, capital_dates, capital_over_time
    
    def get_strategy_summary(self, summary):
        """Get formatted strategy summary"""
        return {
            'total_trades': summary['total_trades'],
            'winning_trades': summary['wins'],
            'losing_trades': summary['losses'],
            'accuracy': summary['accuracy'] * 100,
            'starting_capital': summary['starting_capital'],
            'ending_capital': summary['ending_capital'],
            'net_profit': summary['net_profit'],
            'return_pct': (summary['net_profit'] / summary['starting_capital']) * 100 if summary['starting_capital'] > 0 else 0
        }
