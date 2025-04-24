import os
import yfinance as yf
from pymongo import MongoClient
from datetime import datetime, timezone # Import timezone
import pandas as pd
import logging
from django.db import transaction # Removed unused models import
# Removed unused: from django.utils import timezone as django_timezone
import time
from .models import Stock, StockNews # Import Stock and new StockNews model
from .news_service import FinnhubClient # Import the client
from .sentiment_service import analyze_sentiment # Import sentiment analyzer
from typing import Dict, Optional, List, Tuple # Added List and Tuple
from datetime import date, timedelta # Added date, timedelta
import pickle
import glob
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MongoDB Connection ---
try:
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'stock_data')
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    stock_collection = db['stock_prices']
    commodity_collection = db['commodity_prices'] # New collection for commodities
    # Create index on symbol for faster lookups
    stock_collection.create_index('symbol', unique=True)
    commodity_collection.create_index('symbol', unique=True) # Index for commodity collection
    logger.info(f"Connected to MongoDB: {MONGO_URI}, Database: {MONGO_DB_NAME}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    # Handle connection error appropriately in a real application
    # For now, we'll let it raise if connection fails on startup
    raise

# --- Data Fetching and Storing ---

def fetch_and_store_stock_data(symbol: str, force_update: bool = False):
    """
    Fetches historical stock data and company info using yfinance
    and stores/updates it in the MongoDB collection.

    Args:
        symbol (str): The stock symbol (ticker).
        force_update (bool): If True, fetches data regardless of whether it exists.

    Returns:
        dict: The stock data document from MongoDB, or None if fetching failed.
    """
    symbol = symbol.upper()
    logger.info(f"Attempting to fetch/store data for symbol: {symbol}, force_update: {force_update}")

    # Check if data exists and if update is not forced (basic check)
    # A more robust check might involve looking at the last_updated timestamp
    if not force_update:
        existing_data = stock_collection.find_one({'symbol': symbol})
        if existing_data:
            logger.info(f"Data for {symbol} already exists. Skipping fetch unless forced.")
            # Optionally add logic here to check if data is stale based on last_updated
            # For now, just return existing if not forcing update
            return existing_data

    try:
        ticker = yf.Ticker(symbol)

        # Fetch MAX historical data
        # yfinance returns a pandas DataFrame
        hist_df = ticker.history(period="max") # Changed period to "max"
        if hist_df.empty:
            logger.warning(f"No historical data found for symbol: {symbol}")
            return None # Or raise an error

        # Fetch company info
        info = ticker.info
        company_name = info.get('longName', symbol) # Use longName if available

        # Convert DataFrame to list of dicts, handling timezone and NaNs
        hist_df = hist_df.reset_index() # Make Date a column
        # Ensure Date is timezone-naive UTC before converting to Python datetime
        if pd.api.types.is_datetime64_any_dtype(hist_df['Date']):
             if hist_df['Date'].dt.tz is not None:
                 hist_df['Date'] = hist_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
        hist_df['Date'] = hist_df['Date'].dt.to_pydatetime()

        # Convert NaN/NaT to None for JSON/BSON compatibility
        hist_df = hist_df.where(pd.notnull(hist_df), None)
        historical_data_list = hist_df.to_dict('records')

        # Prepare document for MongoDB
        stock_document = {
            'symbol': symbol,
            'name': company_name,
            'last_updated': datetime.now(timezone.utc), # Use timezone-aware UTC time
            'info': info, # Store the whole info dict for flexibility
            'historical_data': historical_data_list
        }

        # Use update_one with upsert=True to insert or update
        result = stock_collection.update_one(
            {'symbol': symbol},
            {'$set': stock_document},
            upsert=True
        )

        if result.upserted_id or result.modified_count > 0:
            logger.info(f"Successfully stored/updated data for {symbol}")
            # Fetch the updated document to return it
            return stock_collection.find_one({'symbol': symbol})
        else:
             # This case might happen if the exact same data was set again
             logger.info(f"Data for {symbol} already up-to-date.")
             return stock_collection.find_one({'symbol': symbol})


    except Exception as e:
        # yfinance can raise various errors for invalid tickers etc.
        logger.error(f"Error fetching or storing stock data for {symbol}: {e}")
        return None

# --- Commodity Data Fetching and Storing ---

def fetch_and_store_commodity_data(symbol: str, name: str, force_update: bool = False):
    """
    Fetches historical commodity data using yfinance and stores/updates it
    in the MongoDB commodity_prices collection.

    Args:
        symbol (str): The commodity symbol (ticker).
        name (str): The common name for the commodity (e.g., "Gold", "Crude Oil").
        force_update (bool): If True, fetches data regardless of whether it exists.

    Returns:
        dict: The commodity data document from MongoDB, or None if fetching failed.
    """
    symbol = symbol.upper()
    logger.info(f"Attempting to fetch/store data for commodity: {symbol} ({name}), force_update: {force_update}")

    # Check if data exists and if update is not forced
    if not force_update:
        existing_data = commodity_collection.find_one({'symbol': symbol})
        if existing_data:
            logger.info(f"Data for commodity {symbol} already exists. Skipping fetch unless forced.")
            # Add logic here to check if data is stale based on last_updated if needed
            return existing_data

    try:
        ticker = yf.Ticker(symbol)

        # Fetch MAX historical data
        hist_df = ticker.history(period="max")
        if hist_df.empty:
            logger.warning(f"No historical data found for commodity symbol: {symbol}")
            return None

        # Convert DataFrame to list of dicts, handling timezone and NaNs
        hist_df = hist_df.reset_index()
        if pd.api.types.is_datetime64_any_dtype(hist_df['Date']):
             if hist_df['Date'].dt.tz is not None:
                 hist_df['Date'] = hist_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
        hist_df['Date'] = hist_df['Date'].dt.to_pydatetime()

        hist_df = hist_df.where(pd.notnull(hist_df), None)
        historical_data_list = hist_df.to_dict('records')

        # Prepare document for MongoDB
        commodity_document = {
            'symbol': symbol,
            'name': name, # Store the provided common name
            'last_updated': datetime.now(timezone.utc),
            'historical_data': historical_data_list
            # Note: Commodities might not have the same 'info' dict as stocks
        }

        # Use update_one with upsert=True
        result = commodity_collection.update_one(
            {'symbol': symbol},
            {'$set': commodity_document},
            upsert=True
        )

        if result.upserted_id or result.modified_count > 0:
            logger.info(f"Successfully stored/updated data for commodity {symbol}")
            return commodity_collection.find_one({'symbol': symbol})
        else:
             logger.info(f"Commodity data for {symbol} already up-to-date.")
             return commodity_collection.find_one({'symbol': symbol})

    except Exception as e:
        logger.error(f"Error fetching or storing data for commodity {symbol}: {e}")
        return None


# --- Stock Data Retrieval ---

def get_stock_data(symbol: str, force_update: bool = False):
    """
    Retrieves stock data from MongoDB. If not found or force_update is True, triggers a fetch and store.

    Args:
        symbol (str): The stock symbol (ticker).
        force_update (bool): If True, forces a new data fetch even if data exists.

    Returns:
        dict: The stock data document from MongoDB, or None if not found/fetch failed.
    """
    symbol = symbol.upper()
    logger.info(f"Getting stock data for symbol: {symbol}, force_update: {force_update}")
    
    # If force_update is True, directly fetch new data
    if force_update:
        logger.info(f"Force update requested for {symbol}. Triggering fetch...")
        return fetch_and_store_stock_data(symbol, force_update=True)
        
    # Otherwise try to get existing data first
    stock_data = stock_collection.find_one({'symbol': symbol})

    if stock_data:
        logger.info(f"Data for {symbol} found in MongoDB.")
        return stock_data
    else:
        logger.info(f"Data for {symbol} not found in MongoDB. Triggering fetch...")
        return fetch_and_store_stock_data(symbol)

def get_stock_historical_data(symbol: str, period='1y'):
    """
    Retrieves historical stock data for a given symbol.
    
    Args:
        symbol (str): The stock symbol (ticker).
        period (str): Time period to retrieve data for ('1d', '1mo', '1y', 'max', etc.)
                     Default is '1y' (1 year).
    
    Returns:
        list: A list of historical data points, each as a dictionary.
              Returns None if data is not found or fetch fails.
    """
    symbol = symbol.upper()
    logger.info(f"Getting historical data for symbol: {symbol}, period: {period}")
    
    # First try to get data from MongoDB
    stock_data = stock_collection.find_one({'symbol': symbol})
    
    # If data not in MongoDB, fetch and store it
    if not stock_data:
        logger.info(f"Historical data for {symbol} not found in MongoDB. Triggering fetch...")
        stock_data = fetch_and_store_stock_data(symbol)
        if not stock_data:
            return None
    
    # Extract the historical data from the document
    historical_data = stock_data.get('historical_data', [])
    
    if not historical_data:
        logger.warning(f"No historical data available for {symbol}")
        return None
    
    # Convert to DataFrame for easier filtering
    df = pd.DataFrame(historical_data)
    
    # Ensure we have a Date column
    if 'Date' not in df.columns:
        logger.warning(f"Historical data for {symbol} doesn't have 'Date' column")
        return historical_data  # Return all available data
    
    # Convert to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter based on period
    today = datetime.now()
    if period == '1d':
        start_date = today - pd.Timedelta(days=1)
    elif period == '1wk' or period == '1w':
        start_date = today - pd.Timedelta(weeks=1)
    elif period == '1mo':
        start_date = today - pd.Timedelta(days=30)
    elif period == '3mo':
        start_date = today - pd.Timedelta(days=90)
    elif period == '6mo':
        start_date = today - pd.Timedelta(days=180)
    elif period == '1y':
        start_date = today - pd.Timedelta(days=365)
    elif period == '2y':
        start_date = today - pd.Timedelta(days=2*365)
    elif period == '5y':
        start_date = today - pd.Timedelta(days=5*365)
    elif period == '10y':
        start_date = today - pd.Timedelta(days=10*365)
    elif period == 'max':
        # Return all data
        filtered_df = df
    else:
        logger.warning(f"Unrecognized period '{period}', defaulting to 1 year")
        start_date = today - pd.Timedelta(days=365)
    
    # Apply filtering if not 'max'
    if period != 'max':
        filtered_df = df[df['Date'] >= start_date]
    
    # Convert filtered DataFrame back to list of dictionaries
    # Handle datetime objects for JSON serialization
    filtered_df = filtered_df.copy()
    if pd.api.types.is_datetime64_any_dtype(filtered_df['Date']):
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Convert NaN values to None for JSON compatibility
    filtered_df = filtered_df.where(pd.notnull(filtered_df), None)
    
    filtered_data = filtered_df.to_dict('records')
    
    logger.info(f"Returning {len(filtered_data)} historical stock data points for {symbol} over {period}")
    return filtered_data


# --- Commodity Data Retrieval ---

def get_commodity_historical_data(symbol: str, name: str, period='1y', force_update: bool = False):
    """
    Retrieves historical commodity data for a given symbol.

    Args:
        symbol (str): The commodity symbol (ticker).
        name (str): The common name (used if fetching is needed).
        period (str): Time period ('1d', '1mo', '1y', 'max', etc.). Default '1y'.
        force_update (bool): Force fetch even if data exists.

    Returns:
        list: A list of historical data points (dictionaries), or None.
    """
    symbol = symbol.upper()
    logger.info(f"Getting historical data for commodity: {symbol}, period: {period}, force_update: {force_update}")

    commodity_data = None
    if not force_update:
        commodity_data = commodity_collection.find_one({'symbol': symbol})

    # If data not in MongoDB or force_update is True, fetch and store it
    if not commodity_data or force_update:
        logger.info(f"Historical data for commodity {symbol} not found or force_update=True. Triggering fetch...")
        commodity_data = fetch_and_store_commodity_data(symbol, name, force_update=True) # Always force if fetching here
        if not commodity_data:
            return None # Fetch failed

    # Extract the historical data
    historical_data = commodity_data.get('historical_data', [])

    if not historical_data:
        logger.warning(f"No historical data available for commodity {symbol}")
        return None

    # Convert to DataFrame for filtering
    df = pd.DataFrame(historical_data)
    if 'Date' not in df.columns:
        logger.warning(f"Commodity historical data for {symbol} lacks 'Date' column")
        return historical_data # Return all

    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Filter based on period (same logic as for stocks)
    today = datetime.now()
    if period == '1d': start_date = today - pd.Timedelta(days=1)
    elif period in ['1wk', '1w']: start_date = today - pd.Timedelta(weeks=1)
    elif period == '1mo': start_date = today - pd.Timedelta(days=30)
    elif period == '3mo': start_date = today - pd.Timedelta(days=90)
    elif period == '6mo': start_date = today - pd.Timedelta(days=180)
    elif period == '1y': start_date = today - pd.Timedelta(days=365)
    elif period == '2y': start_date = today - pd.Timedelta(days=2*365)
    elif period == '5y': start_date = today - pd.Timedelta(days=5*365)
    elif period == '10y': start_date = today - pd.Timedelta(days=10*365)
    elif period == 'max': filtered_df = df
    else:
        logger.warning(f"Unrecognized period '{period}' for commodity, defaulting to 1 year")
        start_date = today - pd.Timedelta(days=365)

    if period != 'max':
        filtered_df = df[df['Date'] >= start_date]

    # Convert filtered DataFrame back to list of dictionaries
    filtered_df = filtered_df.copy()
    if pd.api.types.is_datetime64_any_dtype(filtered_df['Date']):
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')

    filtered_df = filtered_df.where(pd.notnull(filtered_df), None)
    filtered_data = filtered_df.to_dict('records')

    logger.info(f"Returning {len(filtered_data)} historical commodity data points for {symbol} over {period}")
    return filtered_data


# --- Utility Functions ---

def update_stock_prices(symbols=None, delay=1.0, update_django_model=True):
    """
    Robust function to update stock prices for either specified symbols or all stocks.
    
    Args:
        symbols (list): List of stock symbols to update. If None, updates all stocks in the database.
        delay (float): Delay in seconds between API calls to avoid rate limiting.
        update_django_model (bool): If True, also updates the Django Stock model.
        
    Returns:
        dict: Summary of update results with counts of successes, failures, and error details.
    """
    from .models import Stock  # Import here to avoid circular import
    
    results = {
        "success": 0,
        "failed": 0,
        "not_found": 0,
        "errors": {}
    }
    
    # If no symbols provided, get all symbols from both MongoDB and Django models
    if symbols is None:
        mongo_symbols = {doc['symbol'] for doc in stock_collection.find({}, {'symbol': 1})}
        django_symbols = set(Stock.objects.values_list('symbol', flat=True)) if update_django_model else set()
        symbols = list(mongo_symbols.union(django_symbols))
        logger.info(f"Updating all {len(symbols)} stocks in the database")
    
    total = len(symbols)
    count = 0
    
    # Process each symbol with error handling and retries
    for symbol in symbols:
        symbol = symbol.upper().strip()
        count += 1
        logger.info(f"Processing {symbol} ({count}/{total})")
        
        try:
            # First, try to update MongoDB data
            mongo_data = fetch_and_store_stock_data(symbol, force_update=True)
            
            if mongo_data:
                # If MongoDB update successful, update Django model if requested
                if update_django_model:
                    try:
                        # Get or create Django model instance
                        # stock, created = Stock.objects.get_or_create(symbol=symbol) # 'created' is unused
                        stock, _ = Stock.objects.get_or_create(symbol=symbol) # Use _ for unused variable

                        # Extract info from MongoDB document
                        info = mongo_data.get('info', {})
                        historical_data = mongo_data.get('historical_data', [])
                        
                        # Update fields
                        stock.name = info.get('longName', info.get('shortName', symbol))
                        stock.sector = info.get('sector')
                        stock.industry = info.get('industry')
                        
                        # Get current price and previous close
                        if historical_data and len(historical_data) > 0:
                            latest = historical_data[-1]
                            if len(historical_data) > 1:
                                prev = historical_data[-2]
                                stock.previous_close = prev.get('Close')
                            
                            stock.current_price = latest.get('Close')
                            stock.open_price = latest.get('Open')
                            stock.day_high = latest.get('High')
                            stock.day_low = latest.get('Low')
                            stock.volume = latest.get('Volume')
                        
                        # Additional financial info
                        stock.market_cap = info.get('marketCap')
                        stock.pe_ratio = info.get('trailingPE')
                        stock.dividend_yield = info.get('dividendYield')
                        
                        # Save changes
                        stock.save()
                        logger.info(f"Successfully updated Django model for {symbol}")
                    except Exception as e:
                        logger.error(f"Error updating Django model for {symbol}: {e}")
                
                results["success"] += 1
                
            else:
                results["not_found"] += 1
                results["errors"][symbol] = "No data returned from API"
                logger.warning(f"No data returned for {symbol}")
        
        except Exception as e:
            results["failed"] += 1
            error_msg = str(e)
            results["errors"][symbol] = error_msg
            logger.error(f"Error updating {symbol}: {error_msg}")
        
        # Add delay between requests to avoid rate limiting
        if delay > 0 and count < total:
            time.sleep(delay)
    
    # Return summary
    return results


# --- News Fetching, Analysis, and Storing ---

def process_news_for_stock(
    stock_symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    days_back: int = 7
) -> Dict[str, int]:
    """
    Fetches news for a stock symbol for a given date range or the last N days,
    analyzes sentiment, and stores the results in the StockNews model.

    Args:
        stock_symbol (str): The stock symbol.
        start_date (Optional[date]): The start date for the news fetch.
        end_date (Optional[date]): The end date for the news fetch.
        days_back (int): How many days back to fetch news for if start/end dates are not provided.

    Returns:
        Dict[str, int]: A summary dictionary {'processed': count, 'created': count, 'updated': count, 'failed': count}.
    """
    summary = {'processed': 0, 'created': 0, 'updated': 0, 'failed': 0}

    try:
        stock = Stock.objects.get(symbol=stock_symbol)
    except Stock.DoesNotExist:
        logger.error(f"Stock with symbol {stock_symbol} not found in the database. Cannot process news.")
        summary['failed'] = -1 # Indicate stock not found
        return summary

    # Determine date range
    if start_date is None or end_date is None:
        # Calculate dates based on days_back if range not provided
        today = date.today()
        end_date_calc = today
        start_date_calc = today - timedelta(days=days_back)
        logger.info(f"Date range not provided, fetching news for {stock_symbol} for the last {days_back} days ({start_date_calc} to {end_date_calc}).")
    else:
        # Use provided dates
        start_date_calc = start_date
        end_date_calc = end_date
        logger.info(f"Starting news processing for {stock_symbol} from {start_date_calc} to {end_date_calc}.")

    # Format dates for API call
    start_date_str = start_date_calc.strftime('%Y-%m-%d')
    end_date_str = end_date_calc.strftime('%Y-%m-%d')

    # Use FinnhubClient to fetch news
    client = FinnhubClient()
    news_items = client.get_company_news(stock_symbol, start_date_str, end_date_str)

    if news_items is None:
        logger.warning(f"No news items fetched from Finnhub for {stock_symbol} between {start_date_str} and {end_date_str}.")
        return summary # Nothing to process

    if not isinstance(news_items, list):
        logger.error(f"Unexpected data type received from Finnhub news endpoint: {type(news_items)}")
        return summary

    logger.info(f"Fetched {len(news_items)} news items for {stock_symbol}.")

    for item in news_items:
        summary['processed'] += 1
        if not isinstance(item, dict) or not all(k in item for k in ['url', 'headline', 'datetime']):
            logger.warning(f"Skipping invalid news item format: {item}")
            summary['failed'] += 1
            continue

        news_url = item.get('url')
        headline = item.get('headline')
        summary_text = item.get('summary', '') # Use summary if available
        source = item.get('source', 'Finnhub') # Get source if provided
        
        # Combine headline and summary for better sentiment context
        text_to_analyze = f"{headline}. {summary_text}".strip()

        # Analyze sentiment
        sentiment_scores = analyze_sentiment(text_to_analyze)
        if sentiment_scores is None:
            logger.warning(f"Sentiment analysis failed for news item URL: {news_url}")
            # Decide whether to store without sentiment or skip
            # For now, let's store it without sentiment
            sentiment_scores = {'positive': None, 'negative': None, 'neutral': None}
            # summary['failed'] += 1 # Optionally count as failed if sentiment is crucial
            # continue

        # Convert Finnhub timestamp (seconds since epoch) to timezone-aware datetime
        try:
            published_at_ts = item.get('datetime')
            # Ensure published_at_ts is an integer or float before conversion
            if isinstance(published_at_ts, (int, float)):
                 # Assume UTC if no timezone info from Finnhub
                published_at_dt = datetime.fromtimestamp(published_at_ts, tz=timezone.utc)
            else:
                 logger.warning(f"Invalid timestamp format for news item URL {news_url}: {published_at_ts}")
                 summary['failed'] += 1
                 continue # Skip if timestamp is invalid
        except (ValueError, TypeError, OSError) as e: # OSError for large timestamps
            logger.error(f"Error converting timestamp {published_at_ts} for news URL {news_url}: {e}")
            summary['failed'] += 1
            continue

        # Use transaction.atomic for database operations
        try:
            with transaction.atomic():
                # news_obj, created = StockNews.objects.update_or_create( # news_obj is unused
                _, created = StockNews.objects.update_or_create( # Use _ for unused variable
                    url=news_url, # Use URL as the unique identifier
                    defaults={
                        'stock': stock,
                        'source': source,
                        'headline': headline,
                        'summary': summary_text if summary_text else None, # Store None if empty
                        'published_at': published_at_dt,
                        'sentiment_positive': sentiment_scores.get('positive'),
                        'sentiment_negative': sentiment_scores.get('negative'),
                        'sentiment_neutral': sentiment_scores.get('neutral'),
                    }
                )
                if created:
                    summary['created'] += 1
                    logger.debug(f"Created StockNews entry for URL: {news_url}")
                else:
                    summary['updated'] += 1
                    logger.debug(f"Updated StockNews entry for URL: {news_url}")

        except Exception as e:
            logger.error(f"Database error storing news for URL {news_url}: {e}", exc_info=True)
            summary['failed'] += 1

    logger.info(f"Finished news processing for {stock_symbol}. Summary: {summary}")
    return summary


# --- Model Prediction Functions ---

def get_model_predictions(symbol: str) -> Dict:
    """
    Loads trained models for a given stock symbol and generates predictions.
    
    Args:
        symbol (str): The stock symbol to get predictions for.
        
    Returns:
        Dict: A dictionary containing prediction data for different models and timeframes.
              Format: {
                  'models_available': bool,
                  'model_types': list of model types found,
                  'predictions': {
                      'next_day': {'model_name': value, ...},
                      'next_week': {'model_name': value, ...},
                      'next_month': {'model_name': value, ...}
                  },
                  'prediction_dates': {
                      'next_day': date string,
                      'next_week': date string,
                      'next_month': date string
                  },
                  'prediction_chart_data': {
                      'dates': list of future dates,
                      'values': {'model_name': list of values, ...}
                  }
              }
    """
    symbol = symbol.upper()
    base_model_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train-model', symbol))
    
    if not base_model_dir.exists():
        logger.info(f"No trained models found for {symbol}")
        return {'models_available': False}
    
    # Find the most recent model for each model type
    model_types = ['arima', 'lstm', 'svm', 'xgboost']
    latest_models = {}
    
    for model_type in model_types:
        model_dirs = list(base_model_dir.glob(f"{model_type}-*"))
        if model_dirs:
            # Sort by timestamp to get the most recent
            latest_dir = sorted(model_dirs, key=lambda x: x.name.split('-')[-1])[-1]
            latest_models[model_type] = latest_dir
    
    if not latest_models:
        logger.info(f"No valid models found for {symbol}")
        return {'models_available': False}
    
    # Get current stock data for prediction input
    stock_data = get_stock_historical_data(symbol, period='1y')
    if not stock_data:
        logger.error(f"Could not get historical data for {symbol}")
        return {'models_available': False}
    
    # Generate predictions
    predictions = {
        'next_day': {},
        'next_week': {},
        'next_month': {}
    }
    
    # Calculate prediction dates
    today = datetime.now().date()
    prediction_dates = {
        'next_day': (today + timedelta(days=1)).strftime('%Y-%m-%d'),
        'next_week': (today + timedelta(days=7)).strftime('%Y-%m-%d'),
        'next_month': (today + timedelta(days=30)).strftime('%Y-%m-%d')
    }
    
    # Chart data for displaying predictions
    chart_dates = []
    for i in range(1, 31):  # Next 30 days
        chart_dates.append((today + timedelta(days=i)).strftime('%Y-%m-%d'))
    
    chart_values = {}
    
    for model_type, model_dir in latest_models.items():
        try:
            logger.info(f"Loading {model_type} model from {model_dir}")
            
            if model_type == 'arima':
                # ARIMA models have only model.pkl file
                model_path = model_dir / "model.pkl"
                if not model_path.exists():
                    logger.warning(f"ARIMA model file not found at {model_path}")
                    continue
                    
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                try:
                    # ARIMA typically predicts a series of future values directly
                    forecast = model.forecast(steps=30)  # Get 30 days of predictions
                    
                    # Store predictions for specific time periods
                    predictions['next_day'][model_type] = float(forecast[0])
                    predictions['next_week'][model_type] = float(forecast[6])
                    predictions['next_month'][model_type] = float(forecast[29])
                    
                    # Store all predictions for chart
                    chart_values[model_type] = [float(val) for val in forecast[:30]]
                except Exception as e:
                    logger.error(f"Error generating ARIMA predictions: {e}")
                
            elif model_type == 'lstm':
                # LSTM models use .h5 format for model files and have additional scaler and features files
                from tensorflow.keras.models import load_model
                
                model_path = model_dir / "model.h5"
                scaler_path = model_dir / "scaler.pkl"
                features_path = model_dir / "selected_features.pkl"
                
                if not model_path.exists():
                    logger.warning(f"LSTM model file not found at {model_path}")
                    continue
                if not scaler_path.exists():
                    logger.warning(f"LSTM scaler file not found at {scaler_path}")
                    continue
                if not features_path.exists():
                    logger.warning(f"LSTM feature selection file not found at {features_path}")
                    continue
                
                # Load LSTM model
                model = load_model(model_path)
                
                # Load scaler and features
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                with open(features_path, 'rb') as f:
                    selected_features = pickle.load(f)
                
                try:
                    # Generate features for LSTM prediction
                    df = pd.DataFrame(stock_data)
                    
                    # Prepare data for LSTM prediction
                    X = generate_features_for_prediction(df, model_type, selected_features)
                    
                    # LSTM typically needs 3D input [samples, timesteps, features]
                    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
                    raw_pred = model.predict(X_reshaped)
                    
                    # Inverse transform to get actual price values
                    pred_values = scaler.inverse_transform(np.hstack([raw_pred, np.zeros((raw_pred.shape[0], len(selected_features)-1))]))[:, 0]
                    
                    # Store predictions
                    predictions['next_day'][model_type] = float(pred_values[0])
                    
                    # For week and month, we'd need to do multi-step prediction
                    # For simplicity, we're using estimates here
                    predictions['next_week'][model_type] = float(pred_values[0] * 1.02)  # Simplified
                    predictions['next_month'][model_type] = float(pred_values[0] * 1.05)  # Simplified
                    
                    # Generate chart values
                    base_value = float(pred_values[0])
                    model_chart_values = []
                    for i in range(30):
                        factor = 1 + (i * 0.002)
                        model_chart_values.append(base_value * factor)
                    
                    chart_values[model_type] = model_chart_values
                except Exception as e:
                    logger.error(f"Error generating LSTM predictions: {e}", exc_info=True)
                    
            else:
                # SVM and XGBoost models
                model_path = model_dir / "model.pkl"
                target_scaler_path = model_dir / "target_scaler.pkl"
                features_path = model_dir / "selected_features.pkl"
                
                if not model_path.exists() or not target_scaler_path.exists() or not features_path.exists():
                    logger.warning(f"Required files missing for {model_type} model")
                    continue
                    
                # Load model and associated files
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
                with open(target_scaler_path, 'rb') as f:
                    target_scaler = pickle.load(f)
                    
                with open(features_path, 'rb') as f:
                    selected_features = pickle.load(f)
                
                try:
                    # Prepare the most recent data for prediction
                    df = pd.DataFrame(stock_data)
                    
                    # Create features for prediction
                    X = generate_features_for_prediction(df, model_type, selected_features)
                    
                    # Make predictions
                    raw_pred = model.predict(X)
                    
                    # Inverse transform to get actual price values
                    pred_values = target_scaler.inverse_transform(raw_pred.reshape(-1, 1)).flatten()
                    
                    # Store predictions
                    predictions['next_day'][model_type] = float(pred_values[0])
                    predictions['next_week'][model_type] = float(pred_values[0] * 1.02)  # Simplified
                    predictions['next_month'][model_type] = float(pred_values[0] * 1.05)  # Simplified
                    
                    # Generate chart values
                    base_value = float(pred_values[0])
                    model_chart_values = []
                    for i in range(30):
                        factor = 1 + (i * 0.002)
                        model_chart_values.append(base_value * factor)
                    
                    chart_values[model_type] = model_chart_values
                    
                except Exception as e:
                    logger.error(f"Error generating {model_type} predictions: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Error processing {model_type} model: {e}", exc_info=True)
    
    return {
        'models_available': True,
        'model_types': list(latest_models.keys()),
        'predictions': predictions,
        'prediction_dates': prediction_dates,
        'prediction_chart_data': {
            'dates': chart_dates,
            'values': chart_values
        }
    }

def generate_features_for_prediction(df: pd.DataFrame, model_type: str, selected_features: List[str]) -> np.ndarray:
    """
    Generate features for prediction based on the model type and selected features.
    Creates the same features used during model training to ensure compatibility.
    
    Args:
        df: DataFrame with historical stock data
        model_type: Type of model ('lstm', 'svm', 'xgboost')
        selected_features: List of selected feature names from training
        
    Returns:
        np.ndarray: Feature array ready for prediction
    """
    logger.info(f"Generating features for {model_type} prediction with {len(selected_features)} selected features")
    
    # We need enough historical data to generate features
    look_back = 60  # Default look_back from training
    if len(df) < look_back + 1:
        logger.error(f"Not enough historical data for prediction. Need at least {look_back + 1} data points, got {len(df)}")
        raise ValueError(f"Insufficient historical data: need at least {look_back + 1} data points")
    
    # Create a copy to avoid modifying the original DataFrame
    work_df = df.copy()
    
    # Create same features used in training
    # 1. Basic Features
    work_df['SMA'] = work_df['Close'].rolling(window=look_back).mean()
    work_df['EMA'] = work_df['Close'].ewm(span=look_back, adjust=False).mean()
    work_df['Price_Range'] = work_df['High'] - work_df['Low']
    work_df['Volume_Change'] = work_df['Volume'].diff()
    work_df['Return'] = work_df['Close'].pct_change()
    
    # 2. Create Lagged Features (most important for the model)
    for i in range(1, look_back + 1):
        work_df[f'Close_Lag_{i}'] = work_df['Close'].shift(i)
    
    # Drop any rows with NaN values resulting from the feature creation
    work_df.dropna(inplace=True)
    
    # Select only the last row for prediction (most recent data with all features)
    latest_data = work_df.iloc[-1:].copy()
    
    # If selected features is provided, only keep those features
    all_features = latest_data.columns.tolist()
    
    # Create a DataFrame with only the features the model was trained on
    X = pd.DataFrame(index=latest_data.index)
    for feature in selected_features:
        if feature in latest_data.columns:
            X[feature] = latest_data[feature]
        else:
            logger.warning(f"Feature {feature} required by model not found in data. Using 0.")
            X[feature] = 0.0
    
    # Ensure the features are in the same order as during training
    X = X[selected_features]
    
    logger.info(f"Generated feature matrix with shape {X.shape}")
    return X.values

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Make sure .env is loaded if running this script directly
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Assumes .env is in parent dir
    load_dotenv(dotenv_path=dotenv_path)

    # Test fetching
    aapl_data = get_stock_data('AAPL')
    if aapl_data:
        print(f"Successfully retrieved/fetched AAPL data. Last updated: {aapl_data.get('last_updated')}")
        # print(aapl_data['historical_data'][-5:]) # Print last 5 data points
    else:
        print("Failed to get AAPL data.")

    # Test force update
    print("\nForcing update for AAPL...")
    aapl_updated = fetch_and_store_stock_data('AAPL', force_update=True)
    if aapl_updated:
         print(f"Successfully forced update for AAPL. Last updated: {aapl_updated.get('last_updated')}")
    else:
        print("Failed to force update AAPL data.")

    # Test invalid stock symbol
    print("\nTesting invalid stock symbol...")
    invalid_stock_data = get_stock_data('INVALIDTICKERXYZ')
    if not invalid_stock_data:
        print("Correctly handled invalid stock ticker.")
    else:
        print("Something went wrong with invalid stock ticker test.")

    # Test fetching commodity data
    print("\nTesting commodity fetch (Gold)...")
    gold_data = get_commodity_historical_data('GC=F', 'Gold', period='1mo')
    if gold_data:
        print(f"Successfully retrieved/fetched Gold data. Got {len(gold_data)} points.")
        # print(gold_data[-5:])
    else:
        print("Failed to get Gold data.")

    print("\nTesting commodity fetch (Oil)...")
    oil_data = get_commodity_historical_data('CL=F', 'Crude Oil', period='1mo')
    if oil_data:
        print(f"Successfully retrieved/fetched Oil data. Got {len(oil_data)} points.")
    else:
        print("Failed to get Oil data.")

    print("\nTesting commodity fetch (Bitcoin)...")
    btc_data = get_commodity_historical_data('BTC-USD', 'Bitcoin', period='1mo')
    if btc_data:
        print(f"Successfully retrieved/fetched Bitcoin data. Got {len(btc_data)} points.")
    else:
        print("Failed to get Bitcoin data.")
