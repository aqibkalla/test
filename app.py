import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os

from delta_exchange_api import DeltaExchangeAPI
from trading_strategy import TradingStrategy
from backtester import Backtester

# Page configuration
st.set_page_config(
    page_title="BTCUSD 21 EMA Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 BTCUSD 21 EMA Trading Bot")
st.markdown("Backtesting a 21 EMA strategy on BTCUSD 2-hour candlesticks from Delta Exchange")

# Sidebar configuration
st.sidebar.header("Configuration")

# API Key input
api_key = st.sidebar.text_input(
    "Delta Exchange API Key (optional)", 
    value=os.getenv("DELTA_EXCHANGE_API_KEY", ""),
    type="password",
    help="API key for Delta Exchange (can be left empty for public data)"
)

# Date range selection
end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.now().date(),
    max_value=datetime.now().date()
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=end_date - timedelta(days=30),
    max_value=end_date
)

# Strategy parameters
st.sidebar.subheader("Strategy Parameters")
ema_period = st.sidebar.number_input("EMA Period", value=21, min_value=5, max_value=200)
starting_capital = st.sidebar.number_input("Starting Capital ($)", value=100, min_value=10, max_value=10000)
leverage = st.sidebar.number_input("Leverage", value=50, min_value=1, max_value=100)
risk_pct = st.sidebar.number_input("Risk per Trade (%)", value=2.0, min_value=0.5, max_value=10.0) / 100
target_pct = st.sidebar.number_input("Target per Trade (%)", value=5.0, min_value=1.0, max_value=20.0) / 100
stoploss_points = st.sidebar.number_input("Stop Loss (points)", value=500, min_value=100, max_value=2000)

# Trading fees
st.sidebar.subheader("Delta Exchange Fees")
maker_fee = st.sidebar.number_input("Maker Fee (%)", value=0.02, min_value=0.0, max_value=1.0, step=0.01) / 100
taker_fee = st.sidebar.number_input("Taker Fee (%)", value=0.05, min_value=0.0, max_value=1.0, step=0.01) / 100

# Run backtest button
run_backtest = st.sidebar.button("Run Backtest", type="primary")

@st.cache_data
def fetch_and_process_data(start_date, end_date, api_key):
    """Fetch and process BTCUSD data from Delta Exchange"""
    try:
        api = DeltaExchangeAPI(api_key if api_key else None)
        
        # Calculate number of candles needed (approximate)
        days_diff = (end_date - start_date).days
        limit = max(100, min(1000, days_diff * 12))  # 12 candles per day for 2h timeframe
        
        # Fetch historical data using working implementation
        data = api.fetch_historical_candles(
            symbol="BTCUSD",
            resolution="2h",
            limit=limit
        )
        
        if data is None or len(data) == 0:
            st.error("No data received from Delta Exchange API")
            return None
            
        return data
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_price_chart(df, trades_df=None):
    """Create interactive price chart with EMA and trade signals"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('BTCUSD Price & 21 EMA', 'Trade P&L'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="BTCUSD",
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # 21 EMA line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA21'],
            mode='lines',
            name='21 EMA',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add trade signals if available
    if trades_df is not None and not trades_df.empty:
        # Long entries
        long_entries = trades_df[trades_df['side'] == 'long']
        if not long_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_entries['entry_time'],
                    y=long_entries['entry_price'],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        # Short entries
        short_entries = trades_df[trades_df['side'] == 'short']
        if not short_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_entries['entry_time'],
                    y=short_entries['entry_price'],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
        
        # Add liquidation markers if any
        liquidated_trades = trades_df[trades_df['exit_reason'] == 'Liquidation'] if 'exit_reason' in trades_df.columns else pd.DataFrame()
        if not liquidated_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=liquidated_trades['exit_time'],
                    y=liquidated_trades['exit_price'],
                    mode='markers',
                    name='Liquidations',
                    marker=dict(color='black', size=15, symbol='x')
                ),
                row=1, col=1
            )
        
        # Add liquidation levels if available
        if 'liquidation_price' in trades_df.columns:
            for _, trade in trades_df.iterrows():
                entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
                exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]
                if entry_idx >= 0 and exit_idx >= 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[df.index[entry_idx], df.index[exit_idx]],
                            y=[trade['liquidation_price'], trade['liquidation_price']],
                            mode='lines',
                            name=f"Liquidation Level",
                            line=dict(color='black', width=1, dash='dot'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
        
        # Cumulative P&L
        if 'cumulative_pnl' in trades_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=trades_df['exit_time'],
                    y=trades_df['cumulative_pnl'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title="BTCUSD 21 EMA Strategy Backtest Results",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def display_trade_amounts(trades_df):
    """Display trade amounts and position sizing information"""
    if trades_df is None or trades_df.empty:
        return
    
    st.subheader("💰 Capital Usage & Profit/Loss Analysis")
    
    # Calculate trade amount statistics
    if 'capital_used' in trades_df.columns and 'risk_dollars' in trades_df.columns:
        # Capital usage metrics
        avg_capital_used = trades_df['capital_used'].mean()
        max_capital_used = trades_df['capital_used'].max()
        total_capital_deployed = trades_df['capital_used'].sum()
        
        # Risk metrics
        avg_risk_per_trade = trades_df['risk_dollars'].mean()
        max_risk_per_trade = trades_df['risk_dollars'].max()
        
        # Position value metrics
        if 'position_value' in trades_df.columns:
            avg_position_value = trades_df['position_value'].mean()
            max_position_value = trades_df['position_value'].max()
        else:
            avg_position_value = 0
            max_position_value = 0
        
        # Profit/Loss metrics
        total_pnl = trades_df['pnl'].sum()
        win_trades = trades_df[trades_df['pnl'] > 0]
        lose_trades = trades_df[trades_df['pnl'] < 0]
        
        avg_win_amount = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss_amount = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0
        total_wins = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
        total_losses = lose_trades['pnl'].sum() if len(lose_trades) > 0 else 0
        
        # Fee analysis
        if 'trading_fees' in trades_df.columns:
            total_fees = trades_df['trading_fees'].sum()
            avg_fee_per_trade = trades_df['trading_fees'].mean()
            fee_percentage_of_pnl = (total_fees / abs(total_pnl) * 100) if total_pnl != 0 else 0
        else:
            total_fees = 0
            avg_fee_per_trade = 0
            fee_percentage_of_pnl = 0
        
        # Liquidation analysis
        liquidation_trades = trades_df[trades_df['exit_reason'] == 'Liquidation'] if 'exit_reason' in trades_df.columns else pd.DataFrame()
        liquidation_count = len(liquidation_trades)
        liquidation_rate = (liquidation_count / len(trades_df) * 100) if len(trades_df) > 0 else 0
        liquidation_loss = liquidation_trades['pnl'].sum() if not liquidation_trades.empty else 0
        
        # Display in organized sections
        st.markdown("**Capital Usage**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Capital Used", f"${avg_capital_used:.2f}")
            st.metric("Max Capital Used", f"${max_capital_used:.2f}")
        
        with col2:
            st.metric("Total Capital Deployed", f"${total_capital_deployed:.2f}")
            st.metric("Avg Risk per Trade", f"${avg_risk_per_trade:.2f}")
        
        with col3:
            st.metric("Avg Position Value", f"${avg_position_value:.2f}")
            st.metric("Max Position Value", f"${max_position_value:.2f}")
        
        with col4:
            st.metric("Max Risk per Trade", f"${max_risk_per_trade:.2f}")
            risk_percentage = (avg_risk_per_trade / avg_capital_used * 100) if avg_capital_used > 0 else 0
            st.metric("Risk as % of Capital", f"{risk_percentage:.1f}%")
        
        # Liquidation Risk Analysis
        if liquidation_count > 0:
            st.markdown("**🚨 Liquidation Risk Analysis**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Liquidations", liquidation_count, delta=f"{liquidation_rate:.1f}% of trades")
            
            with col2:
                st.metric("Liquidation Loss", f"${liquidation_loss:.2f}")
            
            with col3:
                avg_liquidation_loss = liquidation_loss / liquidation_count if liquidation_count > 0 else 0
                st.metric("Avg Liquidation Loss", f"${avg_liquidation_loss:.2f}")
            
            with col4:
                st.metric("Liquidation Rate", f"{liquidation_rate:.1f}%")
        else:
            st.markdown("**✅ No Liquidations Detected**")
            st.success("All trades stayed above liquidation levels - good risk management!")
        
        st.info("📊 **Liquidation Formula Updated**: Now using Delta Exchange's official calculation with tiered maintenance margins (0.5% base + 0.075% per BTC above 5 BTC)")
        
        st.markdown("**Profit & Loss Analysis**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total P&L", f"${total_pnl:.2f}", delta=f"{total_pnl:.2f}")
            st.metric("Total Wins", f"${total_wins:.2f}")
        
        with col2:
            st.metric("Total Losses", f"${total_losses:.2f}")
            st.metric("Avg Win", f"${avg_win_amount:.2f}")
        
        with col3:
            st.metric("Avg Loss", f"${avg_loss_amount:.2f}")
            profit_factor = abs(total_wins / total_losses) if total_losses != 0 else float('inf')
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        with col4:
            win_rate = len(win_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
            if avg_loss_amount != 0:
                avg_rr_ratio = abs(avg_win_amount / avg_loss_amount)
                st.metric("Avg R:R Ratio", f"1:{avg_rr_ratio:.2f}")
            else:
                st.metric("Avg R:R Ratio", "∞")
        
        # Trading Fees Analysis
        if total_fees > 0:
            st.markdown("**💸 Trading Fees Analysis**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Fees Paid", f"${total_fees:.2f}")
            
            with col2:
                st.metric("Avg Fee per Trade", f"${avg_fee_per_trade:.2f}")
            
            with col3:
                st.metric("Fees as % of P&L", f"{fee_percentage_of_pnl:.1f}%")
            
            with col4:
                net_after_fees = total_pnl
                gross_before_fees = trades_df['pnl_before_fees'].sum() if 'pnl_before_fees' in trades_df.columns else total_pnl
                st.metric("Gross P&L (before fees)", f"${gross_before_fees:.2f}")
            
            st.info("💸 **Fee Calculation**: Entry and exit fees calculated using Delta Exchange taker rates (0.05% default)")

def display_performance_metrics(trades_df):
    """Display performance metrics in a structured format"""
    if trades_df is None or trades_df.empty:
        st.warning("No trades executed during the backtest period")
        return
    
    # Calculate metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    
    max_drawdown = 0
    if 'cumulative_pnl' in trades_df.columns:
        peak = trades_df['cumulative_pnl'].expanding().max()
        drawdown = (trades_df['cumulative_pnl'] - peak)
        max_drawdown = drawdown.min()
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", total_trades)
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col2:
        st.metric("Winning Trades", winning_trades)
        st.metric("Losing Trades", losing_trades)
    
    with col3:
        st.metric("Total P&L", f"${total_pnl:.2f}")
        st.metric("Average Win", f"${avg_win:.2f}")
    
    with col4:
        st.metric("Average Loss", f"${avg_loss:.2f}")
        st.metric("Max Drawdown", f"${max_drawdown:.2f}")

# Main application logic
if run_backtest:
    if start_date >= end_date:
        st.error("Start date must be before end date")
    else:
        with st.spinner("Fetching data from Delta Exchange..."):
            # Fetch historical data
            historical_data = fetch_and_process_data(start_date, end_date, api_key)
            
            if historical_data is not None:
                st.success(f"Successfully fetched {len(historical_data)} data points")
                
                with st.spinner("Running backtest..."):
                    # Initialize strategy and backtester with capital management
                    strategy = TradingStrategy(
                        ema_period=ema_period,
                        starting_capital=starting_capital,
                        leverage=leverage,
                        risk_pct=risk_pct,
                        target_pct=target_pct,
                        stoploss_points=stoploss_points,
                        maker_fee_pct=maker_fee,
                        taker_fee_pct=taker_fee
                    )
                    
                    backtester = Backtester(strategy)
                    
                    # Run backtest
                    results = backtester.run_backtest(historical_data)
                    
                    if results:
                        df, trades_df = results
                        
                        # Display performance metrics
                        st.header("📊 Performance Summary")
                        display_performance_metrics(trades_df)
                        
                        # Display trade amounts
                        display_trade_amounts(trades_df)
                        
                        # Display interactive chart
                        st.header("📈 Price Chart & Trade Signals")
                        chart = create_price_chart(df, trades_df)
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Display trades table
                        if not trades_df.empty:
                            st.header("📋 Trade Details")
                            
                            # Format trades for display
                            display_trades = trades_df.copy()
                            display_trades['entry_time'] = pd.to_datetime(display_trades['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
                            display_trades['exit_time'] = pd.to_datetime(display_trades['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
                            display_trades['pnl'] = display_trades['pnl'].round(2)
                            display_trades['cumulative_pnl'] = display_trades['cumulative_pnl'].round(2)
                            
                            # Include trade amounts in the display
                            columns_to_show = ['entry_time', 'side', 'entry_price', 'exit_time', 'exit_price', 'pnl', 'cumulative_pnl']
                            if 'capital_used' in display_trades.columns:
                                display_trades['capital_used'] = display_trades['capital_used'].round(2)
                                display_trades['position_value'] = display_trades['position_value'].round(2)
                                display_trades['risk_dollars'] = display_trades['risk_dollars'].round(2)
                                display_trades['units'] = display_trades['units'].round(5)
                                
                                # Add fee information if available
                                if 'trading_fees' in display_trades.columns:
                                    display_trades['trading_fees'] = display_trades['trading_fees'].round(2)
                                    display_trades['entry_fee'] = display_trades['entry_fee'].round(2)
                                    display_trades['exit_fee'] = display_trades['exit_fee'].round(2)
                                    display_trades['pnl_before_fees'] = display_trades['pnl_before_fees'].round(2)
                                
                                # Add liquidation price if available
                                if 'liquidation_price' in display_trades.columns:
                                    display_trades['liquidation_price'] = display_trades['liquidation_price'].round(2)
                                    
                                    # Build columns list based on available data
                                    columns_to_show = ['entry_time', 'side', 'capital_used', 'position_value', 
                                                     'entry_price', 'liquidation_price', 'exit_time', 'exit_price', 'exit_reason']
                                    
                                    if 'trading_fees' in display_trades.columns:
                                        columns_to_show.extend(['pnl_before_fees', 'trading_fees', 'pnl'])
                                    else:
                                        columns_to_show.extend(['pnl'])
                                    
                                    columns_to_show.extend(['risk_dollars', 'units', 'cumulative_pnl'])
                                else:
                                    columns_to_show = ['entry_time', 'side', 'capital_used', 'position_value', 
                                                     'entry_price', 'exit_time', 'exit_price', 'exit_reason']
                                    
                                    if 'trading_fees' in display_trades.columns:
                                        columns_to_show.extend(['pnl_before_fees', 'trading_fees', 'pnl'])
                                    else:
                                        columns_to_show.extend(['pnl'])
                                    
                                    columns_to_show.extend(['risk_dollars', 'units', 'cumulative_pnl'])
                            elif 'units' in display_trades.columns:
                                display_trades['units'] = display_trades['units'].round(5)
                                display_trades['risk_dollars'] = display_trades['risk_dollars'].round(2)
                                columns_to_show.extend(['units', 'risk_dollars'])
                            
                            st.dataframe(
                                display_trades[columns_to_show],
                                use_container_width=True
                            )
                        else:
                            st.info("No trades were generated during the backtest period")
                    else:
                        st.error("Failed to run backtest")

# Initial information display
if not run_backtest:
    st.info("Configure your parameters in the sidebar and click 'Run Backtest' to start the analysis")
    
    # Display strategy explanation
    st.header("📖 Strategy Explanation")
    st.markdown("""
    **21 EMA Strategy for BTCUSD (2-hour timeframe):**
    
    **Long Signal:**
    - Candle closes GREEN and price is above the 21 EMA line
    - Enter long position with capital-based risk management
    
    **Short Signal:**
    - Candle closes RED and price is below the 21 EMA line  
    - Enter short position with capital-based risk management
    
    **Advanced Risk Management:**
    - Starting capital with leverage trading
    - Position sizing based on capital risk percentage
    - Stop loss and targets calculated dynamically
    - Capital compounds with profitable trades
    """)
    
    st.header("🔧 Configuration")
    st.markdown("""
    1. **API Key**: Optional Delta Exchange API key for higher rate limits
    2. **Date Range**: Select the period for backtesting (fetches recent data)
    3. **Strategy Parameters**: 
       - EMA Period: Moving average period
       - Starting Capital: Initial trading capital
       - Leverage: Trading leverage multiplier
       - Risk per Trade: Percentage of capital at risk
       - Target per Trade: Percentage profit target
       - Stop Loss Points: Fixed point stop loss
    4. Click **Run Backtest** to execute the strategy and view results
    """)
