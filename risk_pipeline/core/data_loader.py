"""
Data loader component for RiskPipeline.
"""

import logging
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for downloading and managing financial data.
    
    This class handles downloading stock data from Yahoo Finance,
    caching data locally, and providing data preprocessing functionality.
    """
    
    def __init__(self, cache_dir: str = 'data_cache'):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoader initialized with cache directory: {cache_dir}")
    
    def download_data(self, 
                     symbols: List[str], 
                     start_date: str, 
                     end_date: str,
                     force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download stock data for given symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            force_download: Whether to force re-download even if cached
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        data = {}
        
        for symbol in symbols:
            try:
                symbol_data = self._download_symbol_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    force_download=force_download
                )
                
                if symbol_data is not None and not symbol_data.empty:
                    data[symbol] = symbol_data
                    logger.info(f"Successfully downloaded data for {symbol}: {len(symbol_data)} rows")
                else:
                    logger.warning(f"No data downloaded for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to download data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Downloaded data for {len(data)} symbols")
        return data
    
    def _download_symbol_data(self,
                             symbol: str,
                             start_date: str,
                             end_date: str,
                             force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Download data for a single symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            force_download: Whether to force re-download
            
        Returns:
            DataFrame with stock data or None if failed
        """
        # Check cache first
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.csv"
        
        if not force_download and cache_file.exists():
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Ensure index is datetime after loading from cache
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                
                # Handle timezone-aware datetime conversion
                if hasattr(data.index, 'tz') and data.index.tz is not None:
                    try:
                        # Convert to UTC first, then remove timezone info
                        data.index = data.index.tz_convert('UTC').tz_localize(None)
                        logger.debug(f"Converted timezone-aware index to UTC for {symbol}")
                    except Exception as tz_error:
                        logger.warning(f"Timezone conversion failed for {symbol}: {tz_error}")
                        # Fallback: force UTC conversion
                        try:
                            data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
                            logger.debug(f"Applied UTC fallback for {symbol}")
                        except Exception as fallback_error:
                            logger.error(f"UTC fallback also failed for {symbol}: {fallback_error}")
                            # Last resort: convert to naive datetime
                            data.index = pd.to_datetime(data.index).tz_localize(None)
                            logger.debug(f"Applied naive datetime fallback for {symbol}")
                
                logger.debug(f"Loaded {symbol} data from cache: {len(data)} rows")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached data for {symbol}: {str(e)}")
        
        # Download from Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Clean and standardize data
            data = self._clean_data(data)
            
            # Save to cache
            data.to_csv(cache_file)
            logger.debug(f"Cached {symbol} data: {len(data)} rows")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {str(e)}")
            return None
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize downloaded data.
        
        Args:
            data: Raw downloaded data
            
        Returns:
            Cleaned DataFrame
        """
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index, format='mixed', errors='coerce')
                logger.debug("Converted index to datetime")
            except Exception as e:
                logger.warning(f"Failed to convert index to datetime: {e}")
                # Fallback: try with utc=True
                try:
                    data.index = pd.to_datetime(data.index, utc=True, errors='coerce')
                    logger.debug("Applied UTC fallback")
                except Exception as fallback_error:
                    logger.error(f"UTC fallback also failed: {fallback_error}")
                    raise ValueError("Cannot convert index to datetime")
        
        # Handle timezone-aware datetimes
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            try:
                # Convert to UTC first, then remove timezone info
                data.index = data.index.tz_convert('UTC').tz_localize(None)
                logger.debug("Converted timezone-aware index to UTC")
            except Exception as tz_error:
                logger.warning(f"Timezone conversion failed: {tz_error}")
                # Fallback: force UTC conversion
                try:
                    data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
                    logger.debug("Applied UTC fallback")
                except Exception as fallback_error:
                    logger.error(f"UTC fallback also failed: {fallback_error}")
                    # Last resort: convert to naive datetime
                    data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
                    logger.debug("Applied naive datetime fallback")
        
        # Remove rows with missing values
        data = data.dropna()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in data: {missing_columns}")
            return pd.DataFrame()
        
        # Calculate additional features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate volatility (rolling standard deviation of returns)
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # Remove rows with NaN values after calculations
        data = data.dropna()
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_data_info(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Get information about downloaded data.
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Dictionary with data information
        """
        info = {
            'total_symbols': len(data),
            'symbols': list(data.keys()),
            'date_ranges': {},
            'data_points': {},
            'missing_data': {}
        }
        
        for symbol, df in data.items():
            if not df.empty:
                info['date_ranges'][symbol] = {
                    'start': df.index.min().strftime('%Y-%m-%d'),
                    'end': df.index.max().strftime('%Y-%m-%d'),
                    'days': len(df)
                }
                info['data_points'][symbol] = len(df)
                
                # Check for missing data
                missing_days = self._check_missing_days(df)
                if missing_days > 0:
                    info['missing_data'][symbol] = missing_days
            else:
                info['missing_data'][symbol] = 'No data available'
        
        return info
    
    def _check_missing_days(self, df: pd.DataFrame) -> int:
        """
        Check for missing trading days in the data.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Number of missing days
        """
        if df.empty:
            return 0
        
        # Get business days between start and end
        start_date = df.index.min()
        end_date = df.index.max()
        
        business_days = pd.bdate_range(start=start_date, end=end_date)
        actual_days = df.index
        
        missing_days = len(business_days) - len(actual_days)
        return max(0, missing_days)
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """
        Validate downloaded data quality.
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Dictionary mapping symbols to validation status
        """
        validation_results = {}
        
        for symbol, df in data.items():
            is_valid = True
            issues = []
            
            # Check if data is empty
            if df.empty:
                is_valid = False
                issues.append("Empty DataFrame")
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                is_valid = False
                issues.append(f"Missing columns: {missing_columns}")
            
            # Check for sufficient data points
            if len(df) < 100:
                is_valid = False
                issues.append(f"Insufficient data points: {len(df)}")
            
            # Check for extreme outliers
            if 'Close' in df.columns:
                close_prices = df['Close']
                q1 = close_prices.quantile(0.25)
                q3 = close_prices.quantile(0.75)
                iqr = q3 - q1
                outliers = close_prices[(close_prices < q1 - 1.5 * iqr) | 
                                      (close_prices > q3 + 1.5 * iqr)]
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    is_valid = False
                    issues.append(f"Too many outliers: {len(outliers)}")
            
            validation_results[symbol] = {
                'is_valid': is_valid,
                'issues': issues
            }
            
            if not is_valid:
                logger.warning(f"Data validation failed for {symbol}: {issues}")
        
        return validation_results
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache information
        """
        cache_files = list(self.cache_dir.glob("*.csv"))
        
        info = {
            'cache_directory': str(self.cache_dir),
            'total_files': len(cache_files),
            'total_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            'files': []
        }
        
        for file in cache_files:
            try:
                file_info = {
                    'filename': file.name,
                    'size_mb': file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file.stat().st_mtime)
                }
                info['files'].append(file_info)
            except Exception as e:
                logger.warning(f"Failed to get info for {file}: {str(e)}")
        
        return info
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            symbol: Specific symbol to clear. If None, clears all cache.
        """
        if symbol:
            pattern = f"{symbol}_*.csv"
            cache_files = list(self.cache_dir.glob(pattern))
        else:
            cache_files = list(self.cache_dir.glob("*.csv"))
        
        for file in cache_files:
            try:
                file.unlink()
                logger.debug(f"Deleted cache file: {file}")
            except Exception as e:
                logger.warning(f"Failed to delete {file}: {str(e)}")
        
        logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols with cached data.
        
        Returns:
            List of available symbols
        """
        cache_files = list(self.cache_dir.glob("*.csv"))
        symbols = set()
        
        for file in cache_files:
            # Extract symbol from filename (format: symbol_startdate_enddate.csv)
            symbol = file.stem.split('_')[0]
            symbols.add(symbol)
        
        return sorted(list(symbols)) 