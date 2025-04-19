"""Module containing the Finnhub API client implementation."""
import os
import requests
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# --- Finnhub API Configuration ---
FINNHUB_API_KEYS_VAR = "FINNHUB_API_KEYS"  # Environment variable name
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

class FinnhubClient:
    """Client for interacting with the Finnhub API, handling API key rotation."""
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        if not self.api_keys:
            logger.warning(f"No Finnhub API keys found in environment variable '{FINNHUB_API_KEYS_VAR}'")

    def _load_api_keys(self) -> List[str]:
        """Loads API keys from the environment variable."""
        keys_str = os.getenv(FINNHUB_API_KEYS_VAR)
        if keys_str:
            return [key.strip() for key in keys_str.split(',') if key.strip()]
        return []

    def _get_next_key(self) -> Optional[str]:
        """Rotates to the next available API key."""
        if not self.api_keys:
            return None
        
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key

    def get_company_news(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Fetches company news articles from Finnhub API within the specified date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for news articles
            end_date: End date for news articles
            
        Returns:
            List of news articles
        """
        if not self.api_keys:
            logger.error("No API keys available")
            return []

        params = {
            'symbol': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'token': self._get_next_key()
        }

        try:
            response = requests.get(f"{FINNHUB_BASE_URL}/company-news", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []
