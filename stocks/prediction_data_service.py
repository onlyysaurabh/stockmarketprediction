import pandas as pd
from datetime import datetime, timedelta, date
import logging
from typing import List, Dict, Optional
import numpy as np # Import numpy for NaN handling if needed

# Attempt to import Django-related modules safely
try:
    from django.db import models
    from .services import get_stock_historical_data, get_commodity_historical_data
    from .models import StockNews, Stock
except ImportError:
    # Handle cases where Django models might not be available (e.g., testing setup)
    logging.warning("Could not import Django models/services. Ensure Django is setup.")
    # Define dummy classes or skip functions relying on them if necessary for standalone use
    Stock = None
    StockNews = None
    def get_stock_historical_data(*args, **kwargs): return None
    def get_commodity_historical_data(*args, **kwargs): return None


logger = logging.getLogger(__name__)

# Define commonly used commodities that might affect stock prices
RELEVANT_COMMODITIES = ['GC=F', 'CL=F', '^GSPC']  # Gold, Oil, and S&P 500

def get_prediction_data(stock_symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetches data needed for making predictions for a stock.
    Similar to get_training_data but gets the most recent data for a specific timeframe.
    
    Args:
        stock_symbol (str): The stock symbol to get data for
        days (int): Number of days of historical data to fetch
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with all features ready for prediction, or None if data can't be fetched
    """
    try:
        # Calculate the date range for data collection
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Use the existing training data function since the processing is the same
        return get_training_data(stock_symbol, start_date, end_date)
        
    except Exception as e:
        logger.error(f"Error preparing prediction data for {stock_symbol}: {e}", exc_info=True)
        return None

def get_training_data(stock_symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """
    Fetches and combines all relevant training data for a stock including:
    - Historical stock prices
    - Related commodity prices
    - Sentiment analysis from news
    
    Args:
        stock_symbol (str): The stock symbol to get data for
        start_date (date): Start date for data collection
        end_date (date): End date for data collection
        
    Returns:
        Optional[pd.DataFrame]: Combined DataFrame with all features, or None if data can't be fetched
    """
    try:
        # Get stock historical data
        stock_data = get_stock_historical_data(stock_symbol, start_date, end_date)
        if stock_data is None or stock_data.empty:
            logger.error(f"Could not fetch historical data for {stock_symbol}")
            return None
            
        # Rename stock columns to include symbol
        stock_data = stock_data.add_prefix(f'{stock_symbol}_')
        
        # Create target column (next day's closing price)
        stock_data[f'Target_Close'] = stock_data[f'{stock_symbol}_Close'].shift(-1)
        
        # Get commodity data
        all_data = [stock_data]
        for commodity_symbol in RELEVANT_COMMODITIES:
            commodity_data = get_commodity_historical_data(commodity_symbol, start_date, end_date)
            if commodity_data is not None and not commodity_data.empty:
                # Only keep the Close price for commodities to reduce feature space
                commodity_close = commodity_data['Close'].to_frame(f'{commodity_symbol}_Close')
                all_data.append(commodity_close)
        
        # Get sentiment data if available
        if StockNews:
            news_data = StockNews.objects.filter(
                symbol=stock_symbol,
                date__range=[start_date, end_date]
            ).values('date').annotate(
                avg_sentiment=models.Avg('sentiment_score'),
                news_count=models.Count('id')
            ).order_by('date')
            
            if news_data:
                sentiment_df = pd.DataFrame(news_data)
                sentiment_df.set_index('date', inplace=True)
                all_data.append(sentiment_df)
        
        # Combine all data
        combined_data = pd.concat(all_data, axis=1)
        
        # Handle missing values
        # Forward fill then backward fill to handle gaps
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        
        # Drop rows where target is NA (will be the last row)
        combined_data = combined_data.dropna(subset=[f'Target_Close'])
        
        # Add technical indicators
        combined_data = _add_technical_indicators(combined_data, stock_symbol)
        
        logger.info(f"Successfully prepared training data for {stock_symbol}. Shape: {combined_data.shape}")
        return combined_data
        
    except Exception as e:
        logger.error(f"Error preparing training data for {stock_symbol}: {e}", exc_info=True)
        return None

def _add_technical_indicators(df: pd.DataFrame, stock_symbol: str) -> pd.DataFrame:
    """
    Adds technical indicators to the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data
        stock_symbol (str): Stock symbol for column naming
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Simple Moving Averages
    df[f'{stock_symbol}_SMA_5'] = df[f'{stock_symbol}_Close'].rolling(window=5).mean()
    df[f'{stock_symbol}_SMA_20'] = df[f'{stock_symbol}_Close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    df[f'{stock_symbol}_EMA_5'] = df[f'{stock_symbol}_Close'].ewm(span=5, adjust=False).mean()
    df[f'{stock_symbol}_EMA_20'] = df[f'{stock_symbol}_Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df[f'{stock_symbol}_Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f'{stock_symbol}_RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df[f'{stock_symbol}_Close'].ewm(span=12, adjust=False).mean()
    exp2 = df[f'{stock_symbol}_Close'].ewm(span=26, adjust=False).mean()
    df[f'{stock_symbol}_MACD'] = exp1 - exp2
    df[f'{stock_symbol}_Signal'] = df[f'{stock_symbol}_MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma = df[f'{stock_symbol}_Close'].rolling(window=20).mean()
    std = df[f'{stock_symbol}_Close'].rolling(window=20).std()
    df[f'{stock_symbol}_BB_Upper'] = sma + (std * 2)
    df[f'{stock_symbol}_BB_Lower'] = sma - (std * 2)
    
    # Average True Range (ATR)
    high_low = df[f'{stock_symbol}_High'] - df[f'{stock_symbol}_Low']
    high_close = abs(df[f'{stock_symbol}_High'] - df[f'{stock_symbol}_Close'].shift())
    low_close = abs(df[f'{stock_symbol}_Low'] - df[f'{stock_symbol}_Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df[f'{stock_symbol}_ATR'] = true_range.rolling(window=14).mean()
    
    # Forward fill any NaN values created by indicators
    df = df.fillna(method='ffill')
    
    return df

# Example usage block (requires Django setup or mocking)
if __name__ == '__main__':
    # This block needs Django environment setup to run correctly
    # Example: Set DJANGO_SETTINGS_MODULE environment variable and run django.setup()
    # Or run this script via `python manage.py shell` and import/run functions.

    # --- Setup Django Environment (Example - adjust as needed) ---
    import sys
    import os
    import django
    # Add project root to Python path (adjust relative path if needed)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stockmarketprediction.settings')
    try:
        django.setup()
        # Now re-import models and services after setup
        from stocks.services import get_stock_historical_data, get_commodity_historical_data
        from stocks.models import StockNews, Stock
    except Exception as e:
        print(f"Error setting up Django: {e}. Cannot run example usage.")
        sys.exit(1)
    # --- End Setup ---


    # Load .env variables if needed (e.g., for MONGO_URI in services.py)
    from dotenv import load_dotenv
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)


    test_symbol = 'AAPL' # Use a symbol likely to have data
    print(f"Attempting to get prediction data for: {test_symbol}")
    data = get_prediction_data(test_symbol, days=365)

    if data is not None:
        print(f"\n--- Prediction Data for {test_symbol} (Last 365 days) ---")
        print(data.head())
        print("\n--- Data Info ---")
        data.info()
        print("\n--- Missing Values Check ---")
        print(data.isnull().sum())
        print(f"\nData Shape: {data.shape}")
    else:
        print(f"Failed to get prediction data for {test_symbol}")

    # Test with a symbol that might have less data
    # test_symbol_less = 'MSFT'
    # print(f"\nAttempting to get prediction data for: {test_symbol_less}")
    # data_less = get_prediction_data(test_symbol_less, days=90)
    # if data_less is not None:
    #     print(f"\n--- Prediction Data for {test_symbol_less} (Last 90 days) ---")
    #     print(data_less.head())
    #     print(f"\nData Shape: {data_less.shape}")
    # else:
    #     print(f"Failed to get prediction data for {test_symbol_less}")
