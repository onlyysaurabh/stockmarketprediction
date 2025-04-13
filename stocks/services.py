import os
import yfinance as yf
from pymongo import MongoClient
from datetime import datetime, timezone # Import timezone
import pandas as pd
import logging
from django.db import models
import time

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
                        stock, created = Stock.objects.get_or_create(symbol=symbol)
                        
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
