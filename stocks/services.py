import os
import yfinance as yf
from pymongo import MongoClient
import pandas as pd
import logging
import time
from typing import Dict, Optional
from datetime import datetime, timezone, date, timedelta
from django.db import transaction
from django.utils import timezone

from .models import Stock, StockNews
from .news_client import FinnhubClient
from .sentiment_service import analyze_sentiment

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
    commodity_collection = db['commodity_prices']
    stock_collection.create_index('symbol', unique=True)
    commodity_collection.create_index('symbol', unique=True)
    logger.info(f"Connected to MongoDB: {MONGO_URI}, Database: {MONGO_DB_NAME}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

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

    # Check if data exists and if update is not forced
    if not force_update:
        existing_data = stock_collection.find_one({'symbol': symbol})
        if existing_data:
            logger.info(f"Data for {symbol} already exists. Skipping fetch unless forced.")
            return existing_data

    try:
        ticker = yf.Ticker(symbol)

        # Fetch MAX historical data
        hist_df = ticker.history(period="max")
        if hist_df.empty:
            logger.warning(f"No historical data found for symbol: {symbol}")
            return None

        # Fetch company info
        info = ticker.info
        company_name = info.get('longName', symbol)

        # Convert DataFrame to list of dicts, handling timezone and NaNs
        hist_df = hist_df.reset_index()
        if pd.api.types.is_datetime64_any_dtype(hist_df['Date']):
             if hist_df['Date'].dt.tz is not None:
                 hist_df['Date'] = hist_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
        hist_df['Date'] = hist_df['Date'].dt.to_pydatetime()

        hist_df = hist_df.where(pd.notnull(hist_df), None)
        historical_data_list = hist_df.to_dict('records')

        # Prepare document for MongoDB
        stock_document = {
            'symbol': symbol,
            'name': company_name,
            'last_updated': datetime.now(timezone.utc),
            'info': info,
            'historical_data': historical_data_list
        }

        # Use update_one with upsert=True
        result = stock_collection.update_one(
            {'symbol': symbol},
            {'$set': stock_document},
            upsert=True
        )

        if result.upserted_id or result.modified_count > 0:
            logger.info(f"Successfully stored/updated data for {symbol}")
            return stock_collection.find_one({'symbol': symbol})
        else:
            logger.info(f"Data for {symbol} already up-to-date.")
            return stock_collection.find_one({'symbol': symbol})

    except Exception as e:
        logger.error(f"Error fetching or storing stock data for {symbol}: {e}")
        return None

def fetch_and_store_commodity_data(symbol: str, force_update: bool = False):
    """
    Fetches historical commodity data using yfinance and stores/updates it in the MongoDB collection.
    Common commodity symbols: GC=F (Gold), SI=F (Silver), CL=F (Crude Oil), etc.

    Args:
        symbol (str): The commodity symbol (ticker).
        force_update (bool): If True, fetches data regardless of whether it exists.

    Returns:
        dict: The commodity data document from MongoDB, or None if fetching failed.
    """
    symbol = symbol.upper()
    logger.info(f"Attempting to fetch/store commodity data for symbol: {symbol}, force_update: {force_update}")

    if not force_update:
        existing_data = commodity_collection.find_one({'symbol': symbol})
        if existing_data:
            logger.info(f"Data for commodity {symbol} already exists. Skipping fetch unless forced.")
            return existing_data

    try:
        ticker = yf.Ticker(symbol)

        # Fetch historical data
        hist_df = ticker.history(period="max")
        if hist_df.empty:
            logger.warning(f"No historical data found for commodity: {symbol}")
            return None

        # Convert DataFrame to list of dicts, handling timezone and NaNs
        hist_df = hist_df.reset_index()
        if pd.api.types.is_datetime64_any_dtype(hist_df['Date']):
            if hist_df['Date'].dt.tz is not None:
                hist_df['Date'] = hist_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
        hist_df['Date'] = hist_df['Date'].dt.to_pydatetime()

        # Convert NaN/NaT to None for JSON/BSON compatibility
        hist_df = hist_df.where(pd.notnull(hist_df), None)
        historical_data_list = hist_df.to_dict('records')

        # Prepare document for MongoDB
        commodity_document = {
            'symbol': symbol,
            'name': f"{symbol} Commodity",  # Basic name since commodities don't have company info
            'last_updated': datetime.now(timezone.utc),
            'historical_data': historical_data_list
        }

        # Use update_one with upsert=True to insert or update
        result = commodity_collection.update_one(
            {'symbol': symbol},
            {'$set': commodity_document},
            upsert=True
        )

        if result.modified_count > 0 or result.upserted_id is not None:
            logger.info(f"Successfully updated/inserted commodity data for {symbol}")
            return commodity_document
        else:
            logger.warning(f"No changes made to commodity data for {symbol}")
            return commodity_collection.find_one({'symbol': symbol})

    except Exception as e:
        logger.error(f"Error fetching/storing commodity data for {symbol}: {str(e)}")
        raise

def process_news_for_stock(stock: Stock, start_date: datetime) -> list:
    """
    Fetches and processes news for a given stock, analyzing sentiment and storing in database.
    
    Args:
        stock (Stock): The stock to fetch news for
        start_date (datetime): The date from which to start fetching news
        
    Returns:
        list: List of processed news items
    """
    logger.info(f"Processing news for {stock.symbol} from {start_date}")
    
    try:
        client = FinnhubClient()
        end_date = datetime.now()
        news_items = client.get_company_news(stock.symbol, start_date, end_date)
        
        if not news_items:
            logger.info(f"No news found for {stock.symbol}")
            return []
        
        processed_items = []
        
        for item in news_items:
            # Skip if this news item already exists
            if StockNews.objects.filter(url=item.get('url', '')).exists():
                continue
                
            # Analyze sentiment of the news
            sentiment_score = analyze_sentiment(
                f"{item.get('headline', '')} {item.get('summary', '')}"
            )
            
            # Create news item with sentiment
            news_item = StockNews.objects.create(
                stock=stock,
                title=item.get('headline', ''),
                summary=item.get('summary', ''),
                url=item.get('url', ''),
                source=item.get('source', ''),
                date=datetime.fromtimestamp(item.get('datetime', 0), tz=timezone.utc),
                sentiment_score=sentiment_score
            )
            processed_items.append(news_item)
            
        logger.info(f"Processed {len(processed_items)} new news items for {stock.symbol}")
        return processed_items
        
    except Exception as e:
        logger.error(f"Error processing news for {stock.symbol}: {str(e)}")
        raise

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

def get_commodity_historical_data(symbol: str, name: str = None, period='1y', force_update: bool = False):
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
    if name is None:
        name = f"{symbol} Commodity"
        
    logger.info(f"Getting historical data for commodity: {symbol}, period: {period}, force_update: {force_update}")

    commodity_data = None
    if not force_update:
        commodity_data = commodity_collection.find_one({'symbol': symbol})

    # If data not in MongoDB or force_update is True, fetch and store it
    if not commodity_data or force_update:
        logger.info(f"Historical data for commodity {symbol} not found or force_update=True. Triggering fetch...")
        commodity_data = fetch_and_store_commodity_data(symbol, force_update=True)
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
    if period == '1d': 
        start_date = today - pd.Timedelta(days=1)
    elif period in ['1wk', '1w']: 
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
        filtered_df = df
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
