"""Service for handling stock news operations."""
import logging
from datetime import datetime, timedelta
from typing import List
from .models import Stock, StockNews
from .news_client import FinnhubClient

logger = logging.getLogger(__name__)

def process_news_for_stock(stock: Stock, start_date: datetime, end_date: datetime) -> List[StockNews]:
    """
    Fetches and processes news for a given stock within the specified date range.
    
    Args:
        stock: Stock object
        start_date: Start date for news articles
        end_date: End date for news articles
        
    Returns:
        List of processed StockNews objects
    """
    client = FinnhubClient()
    news_items = client.get_company_news(stock.symbol, start_date, end_date)
    
    processed_items = []
    for item in news_items:
        # Convert timestamp to datetime
        news_date = datetime.fromtimestamp(item.get('datetime', 0))
        
        # Create StockNews object
        news_item = StockNews(
            stock=stock,
            title=item.get('headline', ''),
            summary=item.get('summary', ''),
            source=item.get('source', ''),
            url=item.get('url', ''),
            date=news_date,
            sentiment_score=0.0  # Will be updated by sentiment analysis service
        )
        processed_items.append(news_item)
    
    # Save news items in bulk
    if processed_items:
        StockNews.objects.bulk_create(
            processed_items,
            ignore_conflicts=True
        )
    
    return processed_items

def fetch_stock_news(symbol: str, limit: int = 10):
    """
    Fetches news for a specific stock symbol without storing in the database.
    Useful for displaying news on the frontend.
    
    Args:
        symbol (str): The stock symbol to fetch news for
        limit (int): Maximum number of news items to return
        
    Returns:
        list: List of news items as dictionaries
    """
    logger.info(f"Fetching news for {symbol}, limit: {limit}")
    
    try:
        client = FinnhubClient()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Fetch news from last 30 days by default
        
        news_items = client.get_company_news(symbol, start_date, end_date)
        
        if not news_items:
            logger.info(f"No news found for {symbol}")
            return []
            
        # Format and limit the results
        formatted_news = []
        for item in news_items[:limit]:
            formatted_news.append({
                'headline': item.get('headline', ''),
                'summary': item.get('summary', ''),
                'source': item.get('source', ''),
                'url': item.get('url', ''),
                'date': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'image': item.get('image', '')
            })
        
        return formatted_news
        
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return []
