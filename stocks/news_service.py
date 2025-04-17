import os
import requests
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any # Removed Tuple, Added Any

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# --- Finnhub API Configuration ---
FINNHUB_API_KEYS_VAR = "FINNHUB_API_KEYS" # Environment variable name
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

class FinnhubClient:
    """
    Client for interacting with the Finnhub API, handling API key rotation.
    """
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        if not self.api_keys:
            logger.warning(f"No Finnhub API keys found in environment variable '{FINNHUB_API_KEYS_VAR}'. API calls will fail.")

    def _load_api_keys(self) -> List[str]:
        """Loads API keys from the environment variable."""
        keys_str = os.getenv(FINNHUB_API_KEYS_VAR)
        if keys_str:
            # Split by comma and remove whitespace
            return [key.strip() for key in keys_str.split(',') if key.strip()]
        return []

    def _get_next_key(self) -> Optional[str]:
        """Rotates to the next available API key."""
        if not self.api_keys:
            return None
        
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.debug(f"Using Finnhub API key index: {self.current_key_index}") # Avoid logging the key itself
        return key

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]] | Dict[str, Any]]:
        """
        Makes a request to the Finnhub API, handling key rotation on failure.
        Returns a list of dicts (e.g., for news) or a single dict (e.g., for profile), or None.
        """
        if not self.api_keys:
            logger.error("Cannot make Finnhub request: No API keys configured.")
            return None

        url = f"{FINNHUB_BASE_URL}{endpoint}"
        
        # Try each key until one works or all fail
        initial_key_index = self.current_key_index
        attempts = 0
        max_attempts = len(self.api_keys)

        while attempts < max_attempts:
            api_key = self._get_next_key()
            if not api_key: # Should not happen if initial check passed, but safety first
                 logger.error("Ran out of API keys during rotation attempt.")
                 return None

            headers = {'X-Finnhub-Token': api_key}
            all_params = {**params} # No need to add token here, it's in headers

            try:
                response = requests.get(url, params=all_params, headers=headers, timeout=15) # 15 second timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                
                data = response.json()
                # Finnhub might return an empty list or dict successfully, which is valid
                logger.info(f"Finnhub request successful for endpoint {endpoint}")
                return data

            except requests.exceptions.Timeout:
                logger.warning(f"Finnhub request timed out for endpoint {endpoint}. Key index {self.current_key_index-1} might be slow.")
                # Don't necessarily rotate key on timeout, could be network issue
                # Consider if rotation is desired here. For now, we retry with next key.
            except requests.exceptions.HTTPError as e:
                 logger.warning(f"Finnhub request failed with status {e.response.status_code} for endpoint {endpoint}. Key index {self.current_key_index-1} might be invalid or rate-limited.")
                 # Rotate key on HTTP errors (like 401 Unauthorized, 429 Rate Limit)
            except requests.exceptions.RequestException as e:
                logger.error(f"An unexpected error occurred during Finnhub request: {e}")
                # Rotate key on general request errors too
            
            attempts += 1
            # If we've tried all keys and looped back to the start, stop.
            if self.current_key_index == initial_key_index and attempts > 0:
                 logger.error(f"All Finnhub API keys failed for endpoint {endpoint}.")
                 break # Avoid infinite loop if all keys fail immediately

        logger.error(f"Failed to fetch data from Finnhub endpoint {endpoint} after {attempts} attempts.")
        return None

    def get_company_news(self, symbol: str, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches company news for a given symbol and date range.
        Dates should be in 'YYYY-MM-DD' format.
        Returns a list of news dictionaries or None.
        """
        endpoint = "/company-news"
        params = {
            'symbol': symbol,
            'from': start_date,
            'to': end_date
        }
        logger.info(f"Fetching Finnhub news for {symbol} from {start_date} to {end_date}")
        # Explicitly cast the potential Dict return to List[Dict] or None
        result = self._make_request(endpoint, params)
        if isinstance(result, list):
            return result
        elif result is None:
             return None
        else:
             logger.warning(f"Received unexpected non-list response type for company news: {type(result)}")
             return None


# --- Helper Functions ---

def fetch_stock_news(symbol: str, days_back: int = 7) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches news for a stock symbol for the specified number of days back from today.
    Returns a list of news items (dictionaries) or None if fetching fails.
    """
    client = FinnhubClient()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    news_data = client.get_company_news(symbol, start_date, end_date)
    
    if news_data is None:
        logger.error(f"Failed to fetch any news for {symbol} for the last {days_back} days.")
        return None
        
    # Basic validation/cleaning (optional, depends on API consistency)
    valid_news = []
    for item in news_data:
        if isinstance(item, dict) and item.get('headline') and item.get('url') and item.get('datetime'):
             # Convert timestamp to datetime object
             try:
                 item['published_at_dt'] = datetime.fromtimestamp(item['datetime'])
                 valid_news.append(item)
             except (ValueError, TypeError) as e:
                 logger.warning(f"Could not parse datetime for news item: {item.get('id', 'N/A')}. Error: {e}")
        else:
            logger.warning(f"Skipping invalid news item format: {item}")

    logger.info(f"Successfully fetched {len(valid_news)} valid news items for {symbol}.")
    return valid_news

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Make sure you have a .env file with FINNHUB_API_KEYS="your_key1,your_key2"
    
    test_symbol = 'AAPL' # Example symbol
    print(f"Attempting to fetch news for {test_symbol}...")
    
    news = fetch_stock_news(test_symbol, days_back=7)
    
    if news:
        print(f"\nFetched {len(news)} news items for {test_symbol}:")
        for i, item in enumerate(news[:5]): # Print first 5
            print(f"  {i+1}. Headline: {item.get('headline')}")
            print(f"     Source: {item.get('source')}")
            print(f"     Date: {item.get('published_at_dt')}")
            print(f"     URL: {item.get('url')}")
            print("-" * 20)
    else:
        print(f"\nCould not fetch news for {test_symbol}.")
