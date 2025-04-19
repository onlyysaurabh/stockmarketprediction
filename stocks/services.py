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
from typing import Dict, Optional, Any, List # Added Any, List
from datetime import date, timedelta # Added date, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MongoDB Connection ---
try:
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'stock_data')
    client: MongoClient = MongoClient(MONGO_URI) # type: ignore Pymongo typing can be complex
    db = client[MONGO_DB_NAME] # type: ignore
    stock_collection = db['stock_prices'] # type: ignore
    commodity_collection = db['commodity_prices'] # type: ignore New collection for commodities
    # Create index on symbol for faster lookups
    stock_collection.create_index('symbol', unique=True) # type: ignore
    commodity_collection.create_index('symbol', unique=True) # type: ignore Index for commodity collection
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
        ticker = yf.Ticker(symbol) # type: ignore

        # Fetch MAX historical data
        # yfinance returns a pandas DataFrame
        hist_df = ticker.history(period="max") # type: ignore
        if hist_df.empty:
            logger.warning(f"No historical data found for symbol: {symbol}")
            return None # Or raise an error

        # Fetch company info
        info = ticker.info # type: ignore
        company_name = info.get('longName', symbol) # Use longName if available

        # Convert DataFrame to list of dicts, handling timezone and NaNs
        hist_df = hist_df.reset_index() # Make Date a column
        # Ensure Date is timezone-naive UTC before converting to Python datetime
        if pd.api.types.is_datetime64_any_dtype(hist_df['Date']): # type: ignore
             if hist_df['Date'].dt.tz is not None:
                 hist_df['Date'] = hist_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None) # type: ignore
        hist_df['Date'] = hist_df['Date'].dt.to_pydatetime() # type: ignore

        # Convert NaN/NaT to None for JSON/BSON compatibility
        hist_df = hist_df.where(pd.notnull(hist_df), None) # type: ignore
        historical_data_list = hist_df.to_dict('records') # type: ignore

        # Prepare document for MongoDB
        stock_document: Dict[str, Any] = {
            'symbol': symbol,
            'name': company_name,
            'last_updated': datetime.now(timezone.utc), # Use timezone-aware UTC time
            'info': info, # Store the whole info dict for flexibility
            'historical_data': historical_data_list
        }

        # Use update_one with upsert=True to insert or update
        result = stock_collection.update_one( # type: ignore
            {'symbol': symbol},
            {'$set': stock_document},
            upsert=True
        )

        if result.upserted_id or result.modified_count > 0:
            logger.info(f"Successfully stored/updated data for {symbol}")
            # Fetch the updated document to return it
            return stock_collection.find_one({'symbol': symbol}) # type: ignore
        else:
             # This case might happen if the exact same data was set again
             logger.info(f"Data for {symbol} already up-to-date.")
             return stock_collection.find_one({'symbol': symbol}) # type: ignore


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
        ticker = yf.Ticker(symbol) # type: ignore

        # Fetch MAX historical data
        hist_df = ticker.history(period="max") # type: ignore
        if hist_df.empty:
            logger.warning(f"No historical data found for commodity symbol: {symbol}")
            return None

        # Convert DataFrame to list of dicts, handling timezone and NaNs
        hist_df = hist_df.reset_index()
        if pd.api.types.is_datetime64_any_dtype(hist_df['Date']): # type: ignore
             if hist_df['Date'].dt.tz is not None:
                 hist_df['Date'] = hist_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None) # type: ignore
        hist_df['Date'] = hist_df['Date'].dt.to_pydatetime() # type: ignore

        hist_df = hist_df.where(pd.notnull(hist_df), None) # type: ignore
        historical_data_list = hist_df.to_dict('records') # type: ignore

        # Prepare document for MongoDB
        commodity_document: Dict[str, Any] = {
            'symbol': symbol,
            'name': name, # Store the provided common name
            'last_updated': datetime.now(timezone.utc),
            'historical_data': historical_data_list
            # Note: Commodities might not have the same 'info' dict as stocks
        }

        # Use update_one with upsert=True
        result = commodity_collection.update_one( # type: ignore
            {'symbol': symbol},
            {'$set': commodity_document},
            upsert=True
        )

        if result.upserted_id or result.modified_count > 0:
            logger.info(f"Successfully stored/updated data for commodity {symbol}")
            return commodity_collection.find_one({'symbol': symbol}) # type: ignore
        else:
             logger.info(f"Commodity data for {symbol} already up-to-date.")
             return commodity_collection.find_one({'symbol': symbol}) # type: ignore

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
    stock_data = stock_collection.find_one({'symbol': symbol}) # type: ignore

    if stock_data:
        logger.info(f"Data for {symbol} found in MongoDB.")
        return stock_data
    else:
        logger.info(f"Data for {symbol} not found in MongoDB. Triggering fetch...")
        return fetch_and_store_stock_data(symbol)

# Add type hint for period
def get_stock_historical_data(symbol: str, period: str = '1y') -> Optional[List[Dict[str, Any]]]:
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
    stock_data = stock_collection.find_one({'symbol': symbol}) # type: ignore
    
    # If data not in MongoDB, fetch and store it
    if not stock_data:
        logger.info(f"Historical data for {symbol} not found in MongoDB. Triggering fetch...")
        stock_data = fetch_and_store_stock_data(symbol)
        if not stock_data:
            return None
    
    # Extract the historical data from the document
    historical_data = stock_data.get('historical_data', []) if stock_data else [] # type: ignore
    
    if not historical_data:
        logger.warning(f"No historical data available for {symbol}")
        return None
    
    # Convert to DataFrame for easier filtering
    df = pd.DataFrame(historical_data) # type: ignore
    
    # Ensure we have a Date column
    if 'Date' not in df.columns:
        logger.warning(f"Historical data for {symbol} doesn't have 'Date' column")
        return historical_data  # Return all available data
    
    # Convert to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']): # type: ignore
        df['Date'] = pd.to_datetime(df['Date']) # type: ignore
    
    # Filter based on period
    today = datetime.now()
    start_date: Optional[datetime] = None # Initialize start_date
    filtered_df = df # Default to all data if period is 'max' or unrecognized
    
    if period == '1d': start_date = today - pd.Timedelta(days=1) # type: ignore
    elif period in ['1wk', '1w']: start_date = today - pd.Timedelta(weeks=1) # type: ignore
    elif period == '1mo': start_date = today - pd.Timedelta(days=30) # type: ignore
    elif period == '3mo': start_date = today - pd.Timedelta(days=90) # type: ignore
    elif period == '6mo': start_date = today - pd.Timedelta(days=180) # type: ignore
    elif period == '1y': start_date = today - pd.Timedelta(days=365) # type: ignore
    elif period == '2y': start_date = today - pd.Timedelta(days=2*365) # type: ignore
    elif period == '5y': start_date = today - pd.Timedelta(days=5*365) # type: ignore
    elif period == '10y': start_date = today - pd.Timedelta(days=10*365) # type: ignore
    elif period == 'max': start_date = None # No start date needed for max
    else:
        logger.warning(f"Unrecognized period '{period}', defaulting to 1 year")
        start_date = today - pd.Timedelta(days=365) # type: ignore
    
    # Apply filtering if start_date is set
    if start_date:
        # Ensure 'Date' column is datetime before comparison
        if not pd.api.types.is_datetime64_any_dtype(df['Date']): # type: ignore
             df['Date'] = pd.to_datetime(df['Date']) # type: ignore
        filtered_df = df[df['Date'] >= start_date]
    # else: filtered_df remains the full df

    # Convert filtered DataFrame back to list of dictionaries
    # Handle datetime objects for JSON serialization
    filtered_df = filtered_df.copy() # type: ignore
    if pd.api.types.is_datetime64_any_dtype(filtered_df['Date']): # type: ignore
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Convert NaN values to None for JSON compatibility
    filtered_df = filtered_df.where(pd.notnull(filtered_df), None) # type: ignore
    
    filtered_data: List[Dict[str, Any]] = filtered_df.to_dict('records') # type: ignore
    
    logger.info(f"Returning {len(filtered_data)} historical stock data points for {symbol} over {period}")
    return filtered_data


# --- Commodity Data Retrieval ---

# Add type hint for period
def get_commodity_historical_data(symbol: str, name: str, period: str = '1y', force_update: bool = False) -> Optional[List[Dict[str, Any]]]:
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

    commodity_data: Optional[Dict[str, Any]] = None
    if not force_update:
        commodity_data = commodity_collection.find_one({'symbol': symbol}) # type: ignore

    # If data not in MongoDB or force_update is True, fetch and store it
    if not commodity_data or force_update:
        logger.info(f"Historical data for commodity {symbol} not found or force_update=True. Triggering fetch...")
        commodity_data = fetch_and_store_commodity_data(symbol, name, force_update=True) # Always force if fetching here
        if not commodity_data:
            return None # Fetch failed

    # Extract the historical data
    historical_data = commodity_data.get('historical_data', []) if commodity_data else [] # type: ignore

    if not historical_data:
        logger.warning(f"No historical data available for commodity {symbol}")
        return None

    # Convert to DataFrame for filtering
    df = pd.DataFrame(historical_data) # type: ignore
    if 'Date' not in df.columns:
        logger.warning(f"Commodity historical data for {symbol} lacks 'Date' column")
        return historical_data # Return all

    if not pd.api.types.is_datetime64_any_dtype(df['Date']): # type: ignore
        df['Date'] = pd.to_datetime(df['Date']) # type: ignore

    # Filter based on period (same logic as for stocks)
    today = datetime.now()
    start_date: Optional[datetime] = None # Initialize start_date
    filtered_df = df # Default to all data

    if period == '1d': start_date = today - pd.Timedelta(days=1) # type: ignore
    elif period in ['1wk', '1w']: start_date = today - pd.Timedelta(weeks=1) # type: ignore
    elif period == '1mo': start_date = today - pd.Timedelta(days=30) # type: ignore
    elif period == '3mo': start_date = today - pd.Timedelta(days=90) # type: ignore
    elif period == '6mo': start_date = today - pd.Timedelta(days=180) # type: ignore
    elif period == '1y': start_date = today - pd.Timedelta(days=365) # type: ignore
    elif period == '2y': start_date = today - pd.Timedelta(days=2*365) # type: ignore
    elif period == '5y': start_date = today - pd.Timedelta(days=5*365) # type: ignore
    elif period == '10y': start_date = today - pd.Timedelta(days=10*365) # type: ignore
    elif period == 'max': start_date = None
    else:
        logger.warning(f"Unrecognized period '{period}' for commodity, defaulting to 1 year")
        start_date = today - pd.Timedelta(days=365) # type: ignore

    if start_date:
        # Ensure 'Date' column is datetime before comparison
        if not pd.api.types.is_datetime64_any_dtype(df['Date']): # type: ignore
             df['Date'] = pd.to_datetime(df['Date']) # type: ignore
        filtered_df = df[df['Date'] >= start_date]
    # else: filtered_df remains the full df

    # Convert filtered DataFrame back to list of dictionaries
    filtered_df = filtered_df.copy() # type: ignore
    if pd.api.types.is_datetime64_any_dtype(filtered_df['Date']): # type: ignore
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')

    filtered_df = filtered_df.where(pd.notnull(filtered_df), None) # type: ignore
    filtered_data: List[Dict[str, Any]] = filtered_df.to_dict('records') # type: ignore

    logger.info(f"Returning {len(filtered_data)} historical commodity data points for {symbol} over {period}")
    return filtered_data


# --- Utility Functions ---

# Add type hints for parameters
def update_stock_prices(symbols: Optional[List[str]] = None, delay: float = 1.0, update_django_model: bool = True) -> Dict[str, Any]:
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
    
    results: Dict[str, Any] = {
        "success": 0,
        "failed": 0,
        "not_found": 0,
        "errors": {}
    }
    
    # If no symbols provided, get all symbols from both MongoDB and Django models
    target_symbols: List[str]
    if symbols is None:
        mongo_symbols = {doc['symbol'] for doc in stock_collection.find({}, {'symbol': 1})} # type: ignore
        django_symbols = set(Stock.objects.values_list('symbol', flat=True)) if update_django_model else set()
        target_symbols = list(mongo_symbols.union(django_symbols))
        logger.info(f"Updating all {len(target_symbols)} stocks in the database")
    else:
        target_symbols = symbols # Use provided list
    
    total = len(target_symbols)
    count = 0
    
    # Process each symbol with error handling and retries
    for symbol in target_symbols:
        symbol_upper = symbol.upper().strip()
        count += 1
        logger.info(f"Processing {symbol_upper} ({count}/{total})")
        
        try:
            # First, try to update MongoDB data
            mongo_data = fetch_and_store_stock_data(symbol_upper, force_update=True)
            
            if mongo_data:
                # If MongoDB update successful, update Django model if requested
                if update_django_model:
                    try:
                        # Get or create Django model instance
                        stock, _ = Stock.objects.get_or_create(symbol=symbol_upper) # Use _ for unused variable

                        # Extract info from MongoDB document
                        info = mongo_data.get('info', {}) # type: ignore
                        historical_data = mongo_data.get('historical_data', []) # type: ignore
                        
                        # Update fields
                        stock.name = info.get('longName', info.get('shortName', symbol_upper)) # type: ignore
                        stock.sector = info.get('sector') # type: ignore
                        stock.industry = info.get('industry') # type: ignore
                        
                        # Get current price and previous close
                        if historical_data and len(historical_data) > 0:
                            latest = historical_data[-1]
                            if len(historical_data) > 1:
                                prev = historical_data[-2]
                                stock.previous_close = prev.get('Close') # type: ignore
                            
                            stock.current_price = latest.get('Close') # type: ignore
                            stock.open_price = latest.get('Open') # type: ignore
                            stock.day_high = latest.get('High') # type: ignore
                            stock.day_low = latest.get('Low') # type: ignore
                            stock.volume = latest.get('Volume') # type: ignore
                        
                        # Additional financial info
                        stock.market_cap = info.get('marketCap') # type: ignore
                        stock.pe_ratio = info.get('trailingPE') # type: ignore
                        stock.dividend_yield = info.get('dividendYield') # type: ignore
                        
                        # Save changes
                        stock.save()
                        logger.info(f"Successfully updated Django model for {symbol_upper}")
                    except Exception as e:
                        logger.error(f"Error updating Django model for {symbol_upper}: {e}")
                
                results["success"] += 1
                
            else:
                results["not_found"] += 1
                results["errors"][symbol_upper] = "No data returned from API"
                logger.warning(f"No data returned for {symbol_upper}")
        
        except Exception as e:
            results["failed"] += 1
            error_msg = str(e)
            results["errors"][symbol_upper] = error_msg
            logger.error(f"Error updating {symbol_upper}: {error_msg}")
        
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
) -> Dict[str, Any]: # Return type can include non-ints like -1 for failed
    """
    Fetches news for a stock symbol for a given date range or the last N days,
    analyzes sentiment, and stores the results in the StockNews model.

    Args:
        stock_symbol (str): The stock symbol.
        start_date (Optional[date]): The start date for the news fetch.
        end_date (Optional[date]): The end date for the news fetch.
        days_back (int): How many days back to fetch news for if start/end dates are not provided.

    Returns:
        Dict[str, Any]: A summary dictionary {'processed': count, 'created': count, 'updated': count, 'failed': count}.
    """
    summary: Dict[str, Any] = {'processed': 0, 'created': 0, 'updated': 0, 'failed': 0}

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

    # No need for isinstance check if client guarantees list or None
    # if not isinstance(news_items, list):
    #     logger.error(f"Unexpected data type received from Finnhub news endpoint: {type(news_items)}")
    #     return summary

    logger.info(f"Fetched {len(news_items)} news items for {stock_symbol}.")

    for item in news_items:
        summary['processed'] += 1
        # No need for isinstance check if client guarantees dict items
        # if not isinstance(item, dict) or not all(k in item for k in ['url', 'headline', 'datetime']):
        #     logger.warning(f"Skipping invalid news item format: {item}")
        #     summary['failed'] += 1
        #     continue

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
        published_at_ts = item.get('datetime')
        if published_at_ts is None:
            logger.warning(f"Missing timestamp for news item URL {news_url}")
            summary['failed'] += 1
            continue # Skip if timestamp is missing

        try:
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


# --- Model Training Service (Placeholder) ---

def initiate_model_training(
    stock_symbols: List[str], # Use List from typing
    model_type: str,
    start_date: date,
    end_date: date,
    epochs: int,
    batch_size: int,
    # Add other parameters as needed (e.g., hyperparameters)
) -> Dict[str, Any]: # Added Any import earlier
    """
    Placeholder function to simulate the initiation of model training.
    In a real implementation, this would trigger an asynchronous task (e.g., Celery).
    """
    logger.info(f"--- Received Training Request ---")
    logger.info(f"Symbols: {', '.join(stock_symbols)}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Data Range: {start_date} to {end_date}")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}")
    
    # --- TODO: Implement Actual Training Logic ---
    # 1. Fetch data for symbols between start_date and end_date using get_stock_historical_data
    # 2. Preprocess data (scaling, sequence creation) based on model_type
    # 3. Define and compile the model (LSTM, GRU using TensorFlow/Keras or PyTorch)
    # 4. Train the model
    # 5. Save the trained model (e.g., to a file or database)
    # 6. Log results/metrics
    # ---------------------------------------------

    # Simulate initiation success
    logger.info("Placeholder: Training task successfully initiated (simulation).")
    
    # Return a status dictionary
    return {
        "status": "initiated",
        "message": f"Training initiated for {len(stock_symbols)} stocks with model {model_type}.",
        "symbols": stock_symbols,
        "model_type": model_type,
    }


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
