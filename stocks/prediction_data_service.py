import pandas as pd
from datetime import datetime, timedelta, date
import logging
from typing import List, Dict, Optional
import numpy as np # Import numpy for NaN handling if needed

# Attempt to import Django-related modules safely
try:
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

# Define relevant commodities (can be expanded or made dynamic later)
# Using common symbols from yfinance
RELEVANT_COMMODITIES = {
    'GC=F': 'Gold',
    'CL=F': 'Crude Oil',
    '^GSPC': 'S&P 500', # Example index
    # Add more relevant indices/commodities if needed
}

def get_aggregated_sentiment_for_period(stock_symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetches news sentiment scores for a stock within a date range and aggregates them daily.

    Args:
        stock_symbol (str): The stock symbol.
        start_date (date): The start date.
        end_date (date): The end date.

    Returns:
        pd.DataFrame: DataFrame indexed by date with columns for aggregated sentiment scores
                      (e.g., avg_positive, avg_negative, avg_neutral, news_count).
                      Returns an empty DataFrame if no stock or news found or if Django models unavailable.
    """
    empty_df = pd.DataFrame(columns=['Date', 'avg_positive', 'avg_negative', 'avg_neutral', 'news_count']).set_index('Date')

    if not Stock or not StockNews:
        logger.error("Stock or StockNews model not available. Cannot fetch sentiment.")
        return empty_df

    try:
        stock = Stock.objects.get(symbol=stock_symbol)
        news_items = StockNews.objects.filter(
            stock=stock,
            published_at__date__gte=start_date,
            published_at__date__lte=end_date,
            # Filter out items with null sentiment scores if needed for aggregation quality
            sentiment_positive__isnull=False,
            sentiment_negative__isnull=False,
            sentiment_neutral__isnull=False,
        ).values(
            'published_at',
            'sentiment_positive',
            'sentiment_negative',
            'sentiment_neutral'
        )

        if not news_items.exists():
            logger.info(f"No news items with sentiment scores found for {stock_symbol} between {start_date} and {end_date}")
            return empty_df

        # Convert QuerySet to DataFrame
        news_df = pd.DataFrame(list(news_items))

        # Extract date from datetime
        news_df['Date'] = news_df['published_at'].dt.date

        # Aggregate sentiment scores by date
        # Calculate mean, skipna=True handles days with only NaN scores (though we filter NaNs above)
        sentiment_agg = news_df.groupby('Date').agg(
            avg_positive=('sentiment_positive', 'mean'),
            avg_negative=('sentiment_negative', 'mean'),
            avg_neutral=('sentiment_neutral', 'mean'),
            news_count=('published_at', 'count') # Count number of news items per day
        ).reset_index()

        # Convert Date column to datetime objects for merging
        sentiment_agg['Date'] = pd.to_datetime(sentiment_agg['Date'])
        sentiment_agg.set_index('Date', inplace=True)

        logger.info(f"Aggregated sentiment for {stock_symbol} from {start_date} to {end_date}. Shape: {sentiment_agg.shape}")
        return sentiment_agg

    except Stock.DoesNotExist:
        logger.error(f"Stock {stock_symbol} not found for sentiment aggregation.")
        return empty_df
    except Exception as e:
        logger.error(f"Error aggregating sentiment for {stock_symbol}: {e}", exc_info=True)
        return empty_df


def get_prediction_data(stock_symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetches and prepares data required for stock prediction models over the specified number of past days.

    Args:
        stock_symbol (str): The stock symbol.
        days (int): Number of past days of data to fetch (default: 365).

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing aligned stock prices,
                                 commodity prices, and sentiment scores, indexed by date.
                                 Returns None if essential data (stock prices) cannot be fetched.
    """
    stock_symbol = stock_symbol.upper()
    logger.info(f"Preparing prediction data for {stock_symbol} for the last {days} days.")

    end_date = datetime.now().date()
    # Go back days + buffer to ensure we capture 'days' worth of trading days after filtering
    start_date = end_date - timedelta(days=days + 90) # Add buffer for non-trading days etc.
    target_start_date = end_date - timedelta(days=days) # The actual start date for the final dataset

    # 1. Fetch Stock Historical Data
    # Fetch a longer period initially to ensure we have enough data after filtering non-trading days
    stock_hist_list = get_stock_historical_data(stock_symbol, period='max') # Fetch max, filter later
    if not stock_hist_list:
        logger.error(f"Could not fetch historical stock data for {stock_symbol}.")
        return None

    stock_df = pd.DataFrame(stock_hist_list)
    if 'Date' not in stock_df.columns or 'Close' not in stock_df.columns:
         logger.error(f"Stock data for {stock_symbol} missing required 'Date' or 'Close' columns.")
         return None

    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    # Perform initial date filtering based on the buffered start date
    stock_df = stock_df[stock_df['Date'].dt.date >= start_date].copy()
    if stock_df.empty:
        logger.error(f"No stock data found for {stock_symbol} after initial date filter ({start_date}).")
        return None

    stock_df = stock_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    stock_df.rename(columns={'Close': f'{stock_symbol}_Close',
                             'Open': f'{stock_symbol}_Open',
                             'High': f'{stock_symbol}_High',
                             'Low': f'{stock_symbol}_Low',
                             'Volume': f'{stock_symbol}_Volume'}, inplace=True)
    stock_df.set_index('Date', inplace=True)
    stock_df.sort_index(inplace=True) # Ensure chronological order
    logger.info(f"Fetched and initially filtered stock data for {stock_symbol}. Shape: {stock_df.shape}")


    # 2. Fetch Commodity Historical Data
    all_data_frames = [stock_df]
    for comm_symbol, comm_name in RELEVANT_COMMODITIES.items():
        # Fetch max and filter similarly to stock data
        comm_hist_list = get_commodity_historical_data(comm_symbol, comm_name, period='max')
        if comm_hist_list:
            comm_df = pd.DataFrame(comm_hist_list)
            if 'Date' in comm_df.columns and 'Close' in comm_df.columns:
                comm_df['Date'] = pd.to_datetime(comm_df['Date'])
                comm_df = comm_df[comm_df['Date'].dt.date >= start_date].copy() # Filter date range
                if not comm_df.empty:
                    comm_df = comm_df[['Date', 'Close']].copy()
                    comm_df.rename(columns={'Close': f'{comm_symbol}_Close'}, inplace=True)
                    comm_df.set_index('Date', inplace=True)
                    comm_df.sort_index(inplace=True)
                    all_data_frames.append(comm_df)
                    logger.info(f"Fetched commodity data for {comm_symbol}. Shape: {comm_df.shape}")
                else:
                    logger.warning(f"Commodity data for {comm_symbol} was empty after date filter.")
            else:
                logger.warning(f"Commodity data for {comm_symbol} missing 'Date' or 'Close'. Skipping.")
        else:
            logger.warning(f"Could not fetch commodity data for {comm_symbol} ({comm_name}). Skipping.")

    # 3. Fetch Aggregated Sentiment Data (using the buffered start date)
    sentiment_df = get_aggregated_sentiment_for_period(stock_symbol, start_date, end_date)
    if not sentiment_df.empty:
        all_data_frames.append(sentiment_df)
        logger.info(f"Fetched sentiment data for {stock_symbol}. Shape: {sentiment_df.shape}")
    else:
        logger.warning(f"No sentiment data found or generated for {stock_symbol}.")


    # 4. Merge DataFrames
    # Start with stock_df and left-merge others to keep all trading days
    final_df = stock_df
    for df_to_merge in all_data_frames[1:]: # Skip stock_df itself
        if not df_to_merge.empty:
            # Ensure index is datetime for merging
            if not pd.api.types.is_datetime64_any_dtype(df_to_merge.index):
                 df_to_merge.index = pd.to_datetime(df_to_merge.index)

            final_df = pd.merge(final_df, df_to_merge, left_index=True, right_index=True, how='left')
        # No need to add empty columns explicitly, merge with how='left' handles missing indices

    logger.info(f"Merged all data sources for {stock_symbol}. Shape before final filtering and filling NaNs: {final_df.shape}")

    # 5. Final Date Filtering (apply the target start date)
    final_df = final_df[final_df.index.date >= target_start_date]
    if final_df.empty:
        logger.error(f"DataFrame empty after applying target start date {target_start_date} for {stock_symbol}.")
        return None

    # 6. Handle Missing Values
    # Forward fill is common for time series price/volume data
    price_cols = [col for col in final_df.columns if '_Close' in col or '_Open' in col or '_High' in col or '_Low' in col or '_Volume' in col]
    final_df[price_cols] = final_df[price_cols].ffill()

    # For sentiment, missing values mean no news on that trading day. Fill with 0s.
    sentiment_cols = ['avg_positive', 'avg_negative', 'avg_neutral', 'news_count']
    for col in sentiment_cols:
        if col in final_df.columns:
            # Ensure the column exists before trying to fillna
            final_df[col].fillna(0, inplace=True)
        else:
            # If sentiment data was completely missing, add the columns with 0s
            final_df[col] = 0.0 if col != 'news_count' else 0

    # Drop any remaining rows with NaNs (e.g., at the very beginning if ffill couldn't fill stock/commodity prices)
    initial_rows = len(final_df)
    final_df.dropna(inplace=True)
    if len(final_df) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(final_df)} rows with NaNs after filling for {stock_symbol}.")

    if final_df.empty:
        logger.error(f"Final DataFrame is empty after processing and NaN handling for {stock_symbol}.")
        return None

    # Ensure data types are appropriate (e.g., float for prices/sentiment, int for volume/count)
    for col in final_df.columns:
        if 'Volume' in col or 'count' in col:
            final_df[col] = final_df[col].astype(int)
        else:
            final_df[col] = final_df[col].astype(float)


    logger.info(f"Prepared prediction data for {stock_symbol}. Final Shape: {final_df.shape}")
    # print(final_df.head()) # For debugging
    # print(final_df.isnull().sum()) # Check for remaining NaNs

    return final_df

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
