# BTCUSD 21 EMA Trading Bot

## Overview

A Streamlit-based trading bot application that implements a 21 EMA (Exponential Moving Average) strategy for BTCUSD trading. The system fetches historical market data from Delta Exchange, applies technical analysis using EMA indicators, and provides backtesting capabilities to evaluate strategy performance. The application features an interactive web interface for configuration, real-time data visualization with Plotly charts, and comprehensive trade analysis including P&L calculations and performance metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Web-based interface providing interactive controls and real-time data visualization
- **Plotly Integration**: Advanced charting library for candlestick charts, EMA overlays, and performance graphs
- **Sidebar Configuration**: User inputs for API keys, date ranges, and strategy parameters

### Backend Architecture
- **Modular Design**: Separated into distinct components for API handling, strategy logic, and backtesting
- **Strategy Engine**: Implements 21 EMA trading logic with configurable parameters for stop-loss and profit targets
- **Backtesting Engine**: Processes historical data to simulate trades and calculate performance metrics
- **Data Pipeline**: Fetches, processes, and transforms market data for analysis

### Core Strategy Logic
- **21 EMA Signal Generation**: Identifies long signals when green candles touch EMA, short signals when red candles touch EMA
- **Risk Management**: Configurable stop-loss and profit target points
- **Trade Execution**: Simulated position management with entry/exit tracking

### Data Processing
- **Real-time Data Fetching**: Integration with Delta Exchange API for historical candlestick data
- **Technical Indicators**: EMA calculations and candle pattern analysis
- **Performance Analytics**: P&L calculations, cumulative returns, and trade statistics

## External Dependencies

### Market Data Provider
- **Delta Exchange API**: Primary source for BTCUSD historical market data and real-time price feeds
- **API Authentication**: Optional API key support for enhanced data access

### Python Libraries
- **Streamlit**: Web application framework for the user interface
- **Plotly**: Interactive charting and data visualization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations for technical indicators
- **Requests**: HTTP client for API communications

### Data Requirements
- **OHLCV Data**: Open, High, Low, Close, Volume candlestick data
- **2-hour Timeframe**: Default resolution for strategy implementation
- **Historical Range**: Configurable date ranges for backtesting periods