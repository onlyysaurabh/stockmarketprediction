"""
MongoDB utility functions for the stocks app.
"""
import os
import pymongo
from datetime import datetime
from django.conf import settings

def get_mongo_client():
    """
    Get a MongoDB client using connection info from settings.
    """
    mongo_uri = settings.MONGO_URI
    return pymongo.MongoClient(mongo_uri)

def get_mongo_db():
    """
    Get the MongoDB database using db name from settings.
    """
    client = get_mongo_client()
    return client[settings.MONGO_DB_NAME]

# Collection name constants
STOCKS_COLLECTION = 'stocks'
STOCK_PRICES_COLLECTION = 'stock_prices'
COMMODITY_PRICES_COLLECTION = 'commodity_prices' 
STOCK_NEWS_COLLECTION = 'stock_news'
WATCHLISTS_COLLECTION = 'watchlists'
WATCHLIST_ITEMS_COLLECTION = 'watchlist_items'

# --- Stock CRUD operations ---

def save_stock(stock_data):
    """
    Save stock information to MongoDB.
    """
    db = get_mongo_db()
    collection = db[STOCKS_COLLECTION]
    
    # Check if this stock already exists
    existing = collection.find_one({'symbol': stock_data['symbol']})
    if existing:
        # Update the existing document
        stock_data['updated_at'] = datetime.now()
        collection.update_one(
            {'_id': existing['_id']},
            {'$set': stock_data}
        )
        return existing['_id']
    else:
        # Insert new document
        stock_data['created_at'] = datetime.now()
        stock_data['updated_at'] = datetime.now()
        return collection.insert_one(stock_data).inserted_id

def get_stock_by_symbol(symbol):
    """
    Get stock information from MongoDB by symbol.
    """
    db = get_mongo_db()
    collection = db[STOCKS_COLLECTION]
    return collection.find_one({'symbol': symbol})

# --- Stock Price operations ---

def save_stock_price(price_data):
    """
    Save stock price data to MongoDB.
    """
    db = get_mongo_db()
    collection = db[STOCK_PRICES_COLLECTION]
    
    # Check if we already have price data for this symbol/date combination
    existing = collection.find_one({
        'symbol': price_data['symbol'],
        'date': price_data['date']
    })
    
    if existing:
        # Update existing document
        collection.update_one(
            {'_id': existing['_id']},
            {'$set': price_data}
        )
        return existing['_id']
    else:
        # Insert new document
        price_data['created_at'] = datetime.now()
        return collection.insert_one(price_data).inserted_id

# --- Commodity Price operations ---

def save_commodity_price(price_data):
    """
    Save commodity price data to MongoDB.
    """
    db = get_mongo_db()
    collection = db[COMMODITY_PRICES_COLLECTION]
    
    # Check if we already have price data for this commodity/date combination
    existing = collection.find_one({
        'symbol': price_data['symbol'],
        'date': price_data['date']
    })
    
    if existing:
        # Update existing document
        collection.update_one(
            {'_id': existing['_id']},
            {'$set': price_data}
        )
        return existing['_id']
    else:
        # Insert new document
        price_data['created_at'] = datetime.now()
        return collection.insert_one(price_data).inserted_id

# --- News operations ---

def save_stock_news(news_data):
    """
    Save stock news data to MongoDB.
    """
    db = get_mongo_db()
    collection = db[STOCK_NEWS_COLLECTION]
    
    # Check if this news article already exists using the url as a unique identifier
    existing = collection.find_one({'url': news_data['url']})
    if existing:
        # Update the existing document
        collection.update_one(
            {'_id': existing['_id']},
            {'$set': news_data}
        )
        return existing['_id']
    else:
        # Insert new document
        news_data['created_at'] = datetime.now()
        return collection.insert_one(news_data).inserted_id

def get_stock_news(symbol=None, limit=50):
    """
    Get stock news from MongoDB, optionally filtered by stock symbol.
    """
    db = get_mongo_db()
    collection = db[STOCK_NEWS_COLLECTION]
    
    query = {'stock_symbol': symbol} if symbol else {}
    
    # Sort by published_at in descending order (newest first)
    return list(collection.find(query).sort('published_at', pymongo.DESCENDING).limit(limit))

# --- Watchlist operations ---

def save_watchlist(watchlist_data):
    """
    Save watchlist to MongoDB.
    Note: This should be linked to a Django User model via user_id
    """
    db = get_mongo_db()
    collection = db[WATCHLISTS_COLLECTION]
    
    # Check if this user already has a watchlist
    existing = collection.find_one({'user_id': watchlist_data['user_id']})
    if existing:
        # Update the existing document
        watchlist_data['updated_at'] = datetime.now()
        collection.update_one(
            {'_id': existing['_id']},
            {'$set': watchlist_data}
        )
        return existing['_id']
    else:
        # Insert new document
        watchlist_data['created_at'] = datetime.now()
        watchlist_data['updated_at'] = datetime.now()
        return collection.insert_one(watchlist_data).inserted_id

def get_watchlist_by_user_id(user_id):
    """
    Get a user's watchlist from MongoDB.
    """
    db = get_mongo_db()
    collection = db[WATCHLISTS_COLLECTION]
    return collection.find_one({'user_id': user_id})

# --- Watchlist Item operations ---

def save_watchlist_item(item_data):
    """
    Save watchlist item to MongoDB.
    """
    db = get_mongo_db()
    collection = db[WATCHLIST_ITEMS_COLLECTION]
    
    # Check if this watchlist item already exists
    existing = collection.find_one({
        'watchlist_id': item_data['watchlist_id'],
        'symbol': item_data['symbol']
    })
    
    if existing:
        # Update the existing document
        item_data['updated_at'] = datetime.now()
        collection.update_one(
            {'_id': existing['_id']},
            {'$set': item_data}
        )
        return existing['_id']
    else:
        # Insert new document
        item_data['created_at'] = datetime.now()
        item_data['updated_at'] = datetime.now()
        return collection.insert_one(item_data).inserted_id

def get_watchlist_items(watchlist_id):
    """
    Get all watchlist items for a specific watchlist.
    """
    db = get_mongo_db()
    collection = db[WATCHLIST_ITEMS_COLLECTION]
    return list(collection.find({'watchlist_id': watchlist_id}))

def remove_watchlist_item(watchlist_id, symbol):
    """
    Remove an item from a watchlist.
    """
    db = get_mongo_db()
    collection = db[WATCHLIST_ITEMS_COLLECTION]
    result = collection.delete_one({
        'watchlist_id': watchlist_id,
        'symbol': symbol
    })
    return result.deleted_count > 0