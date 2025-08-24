import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import streamlit as st

class DeltaExchangeAPI:
    """Delta Exchange API client for fetching historical market data"""
    
    def __init__(self, api_key=None):
        self.base_url = "https://api.india.delta.exchange"
        self.api_key = api_key
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
    
    def fetch_historical_candles(self, symbol='BTCUSD', resolution='2h', limit=500):
        """
        Fetch historical candlestick data from Delta Exchange using working implementation
        
        Args:
            symbol (str): Trading pair symbol (default: BTCUSD)
            resolution (str): Timeframe (1m, 2m, 5m, 15m, 30m, 1h, 2h, 4h, 1d)
            limit (int): Number of candles to fetch
            
        Returns:
            pandas.DataFrame: Historical OHLCV data with time as index
        """
        try:
            url = f'{self.base_url}/v2/history/candles'
            
            # Interval mapping from your working code
            interval_map = {
                '1m': 60, '2m': 120, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '2h': 7200, '4h': 14400, '1d': 86400
            }
            
            interval_seconds = interval_map.get(resolution)
            if interval_seconds is None:
                raise ValueError(f"Unsupported resolution: {resolution}")
            
            # Calculate start and end times based on your working implementation
            end = int(time.time())
            start = end - (interval_seconds * limit)
            
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'start': start,
                'end': end,
                'limit': limit
            }
            
            resp = requests.get(url, params=params)
            data = resp.json()
            
            if 'result' not in data or data['result'] is None:
                raise Exception(f"Error fetching data: {data}")
            
            # Process data exactly like your working code
            df = pd.DataFrame(data['result'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.sort_values('time')
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Set time as index like your working implementation
            df.set_index('time', inplace=True)
            
            # Add timestamp column for compatibility with existing code
            df['timestamp'] = df.index
            
            st.success(f"Successfully fetched {len(df)} candles from Delta Exchange")
            return df
            
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
            return self._create_fallback_data(limit)
    
    def _get_product_id(self, symbol):
        """Get product ID for a given symbol"""
        try:
            url = f"{self.base_url}/v2/products"
            response = self._make_request(url)
            
            if response and 'result' in response:
                products = response['result']
                for product in products:
                    if product.get('symbol') == symbol:
                        return product.get('id')
            return None
            
        except Exception:
            return None
    
    def _make_request(self, url, params=None):
        """Make HTTP request with error handling and rate limiting"""
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                st.warning("Rate limit reached, waiting 60 seconds...")
                time.sleep(60)
                response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API request failed with status code: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return None
    
    def _create_fallback_data(self, limit=500):
        """Create fallback data when API fails"""
        try:
            # Generate realistic BTCUSD price data for demonstration
            end_time = int(time.time())
            start_time = end_time - (7200 * limit)  # 2 hours * limit
            
            # Create time series (2-hour intervals)
            timestamps = []
            current_time = start_time
            while current_time < end_time:
                timestamps.append(current_time)
                current_time += 7200  # 2 hours in seconds
            
            # Generate realistic price data
            base_price = 95000  # Current BTC price range
            num_points = len(timestamps)
            
            # Generate price walk
            returns = np.random.normal(0, 0.02, num_points)  # 2% volatility
            prices = [base_price]
            
            for i in range(1, num_points):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, 1000))  # Minimum price floor
            
            # Create OHLCV data
            data = []
            for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
                # Generate realistic OHLC from close price
                volatility = abs(np.random.normal(0, 0.01))  # 1% intrabar volatility
                
                high = close * (1 + volatility)
                low = close * (1 - volatility)
                
                if i > 0:
                    open_price = prices[i-1]
                else:
                    open_price = close
                
                # Ensure OHLC relationships are valid
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                volume = np.random.uniform(100, 1000)  # Random volume
                
                data.append({
                    'time': timestamp,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': round(volume, 2)
                })
            
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.sort_values('time')
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Set time as index
            df.set_index('time', inplace=True)
            df['timestamp'] = df.index
            
            st.info("Using simulated data due to API limitations")
            return df
            
        except Exception as e:
            st.error(f"Failed to create fallback data: {str(e)}")
            return None
