import csv
import os
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpRequest, Http404 # Keep HttpRequest for type hinting
from django.views.decorators.http import require_GET # require_POST removed as unused
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError
from .models import Watchlist, WatchlistItem, Stock, StockNews # Import Stock and StockNews
from .forms import WatchlistItemForm, UserUpdateForm
# Import specific functions needed from services
from .services import (
    get_stock_data, 
    get_stock_historical_data, 
    get_commodity_historical_data, 
    process_news_for_stock,
    get_model_predictions
)
from .ai_service import get_ai_stock_analysis  # Import the new AI analysis function
import logging
import json
import yfinance as yf
from decimal import Decimal
from datetime import datetime, timedelta, date  # Added date import here
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm # Added PasswordChangeForm
from django.contrib.auth import login, update_session_auth_hash, logout
from django.utils import timezone # Import timezone for date calculations
from .news_service import fetch_stock_news

logger = logging.getLogger(__name__)

def home(request: HttpRequest):
    """Home view with search functionality."""
    # Check if search query exists
    query = request.GET.get('query')
    if query:
        # Redirect to stock detail page if search query exists
        return redirect('stock_detail', symbol=query.upper())
        
    return render(request, 'stocks/home.html')

@require_GET
def search_suggestions_view(request: HttpRequest):
    """Return JSON list of stock symbols matching the search term for autocomplete."""
    search_term = request.GET.get('term', '').upper()
    if not search_term:
        return JsonResponse([], safe=False)
    
    # Path to the CSV file containing stock symbols
    csv_path = os.path.join(settings.BASE_DIR, 'stock_symbols.csv')
    
    suggestions = []
    try:
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                symbol = row.get('Symbol', '')
                name = row.get('Name', '')
                
                # Check if the search term matches the beginning of the symbol or is in the name
                if (symbol and symbol.upper().startswith(search_term)) or \
                   (name and search_term in name.upper()):
                    suggestions.append({
                        'value': symbol,
                        'label': f"{symbol} - {name}" if name else symbol
                    })
                    
                    # Limit to 10 suggestions for performance
                    if len(suggestions) >= 10:
                        break
    except Exception as e:
        logger.error(f"Error reading stock symbols file: {e}")
        
    return JsonResponse(suggestions, safe=False)

def register_view(request: HttpRequest):
    """Handle user registration."""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful. You are now logged in.")
            return redirect('home') # Redirect to home page after successful registration
        else:
            # Pass the form with errors back to the template
            messages.error(request, "Registration unsuccessful. Please correct the errors below.")
    else:
        form = UserCreationForm() # Create a blank form for GET request
    
    # Render the registration page with the form
    # Ensure the template path matches your project structure
    return render(request, 'registration/register.html', {'form': form})

def stock_detail(request: HttpRequest, symbol: str):
    """Display detailed information about a specific stock."""

    # Uppercase the symbol for consistency
    symbol = symbol.upper()

    # Get the Stock object from Django DB first
    try:
        stock_obj = Stock.objects.get(symbol=symbol)
    except Stock.DoesNotExist:
        # Optionally try fetching from MongoDB/yfinance if not in Django DB yet
        # For now, assume if it's not in Stock model, we can't show details
        messages.error(request, f"Stock symbol {symbol} not found in our database.")
        return redirect('home')

    # Fetch the comprehensive data from the service (MongoDB)
    # This might be redundant if Stock model is kept up-to-date, but good for extra info
    stock_data = get_stock_data(symbol)

    if not stock_data:
        # We have the stock_obj, but maybe MongoDB data is missing? Log warning.
        logger.warning(f"Could not retrieve detailed data (MongoDB) for stock symbol {symbol}, but Stock object exists.")
        stock_info = {} # Use empty dict if MongoDB data fails
    else:
        # The 'info' dictionary is inside stock_data
        stock_info = stock_data.get('info', {}) # Get the info dict, default to empty if missing
    
    # Get stock historical data for charting - use 'max' to get all available data
    # Note: get_stock_historical_data also fetches from MongoDB if needed
    stock_historical_data = get_stock_historical_data(symbol, period='max') 
    
    # Prepare stock data for Chart.js
    chart_dates = []
    chart_close = []
    chart_open = []
    chart_high = []
    chart_low = []
    chart_volume = []
    
    if stock_historical_data: # Check if data was successfully fetched
        for data in stock_historical_data:
            # Check if Date is already a string or datetime object
            if isinstance(data.get("Date"), str):
                chart_dates.append(data["Date"])  # Date is already a formatted string
            elif isinstance(data.get("Date"), datetime):
                 chart_dates.append(data["Date"].strftime("%Y-%m-%d")) # Convert datetime
            # Handle cases where Date might be missing or None if necessary
                
            chart_close.append(data.get("Close"))
            chart_open.append(data.get("Open"))
            chart_high.append(data.get("High"))
            chart_low.append(data.get("Low"))
            chart_volume.append(data.get("Volume"))
    else:
        messages.warning(request, f"Could not retrieve historical data for {symbol}.")


    # --- Fetch Commodity Data ---
    commodities_to_fetch = {
        'GC=F': 'Gold',
        'CL=F': 'Crude Oil',
        'BTC-USD': 'Bitcoin',
    }
    commodity_chart_data = {}

    for comm_symbol, comm_name in commodities_to_fetch.items():
        comm_hist_data = get_commodity_historical_data(comm_symbol, comm_name, period='max')
        if comm_hist_data:
            comm_dates = []
            comm_close = []
            for data in comm_hist_data:
                 # Ensure date formatting consistency
                if isinstance(data.get("Date"), str):
                    comm_dates.append(data["Date"])
                elif isinstance(data.get("Date"), datetime):
                    comm_dates.append(data["Date"].strftime("%Y-%m-%d"))
                comm_close.append(data.get("Close"))
            
            commodity_chart_data[comm_name] = {
                'dates': json.dumps(comm_dates),
                'close': json.dumps(comm_close)
            }
    else:
         messages.warning(request, f"Could not retrieve historical data for {comm_name} ({comm_symbol}).")

    # --- Fetch News Data and Calculate Sentiment ---
    seven_days_ago = timezone.now() - timedelta(days=7)
    news_items = StockNews.objects.filter(
        stock=stock_obj,
        published_at__gte=seven_days_ago
    ).order_by('-published_at') # Get news for the last 7 days
    
    # If no news is available for the past 7 days, try to fetch and store it
    if not news_items.exists():
        today = date.today()
        seven_days_ago_date = today - timedelta(days=7)
        
        logger.info(f"No recent news found for {symbol}. Attempting to fetch news for the past 7 days.")
        try:
            # Process news for this stock with date range of the last 7 days
            result = process_news_for_stock(symbol, start_date=seven_days_ago_date, end_date=today)
            
            if result.get('created', 0) > 0 or result.get('updated', 0) > 0:
                messages.success(request, f"Successfully fetched {result.get('created', 0) + result.get('updated', 0)} news articles for {symbol}.")
                # Fetch the newly added news
                news_items = StockNews.objects.filter(
                    stock=stock_obj,
                    published_at__gte=seven_days_ago
                ).order_by('-published_at')
            elif result.get('processed', 0) == 0:
                messages.info(request, f"No news articles found for {symbol} in the past 7 days.")
            else:
                messages.warning(request, f"Attempted to fetch news for {symbol} but found none or encountered issues.")
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            messages.error(request, f"Could not fetch news for {symbol}. Please try again later.")
    
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    total_analyzed = 0

    for news in news_items:
        # Check if sentiment scores exist
        pos = news.sentiment_positive
        neg = news.sentiment_negative
        neu = news.sentiment_neutral

        # Ensure scores are not None before comparison
        if pos is not None and neg is not None and neu is not None:
            total_analyzed += 1
            if pos > neg and pos > neu:
                sentiment_counts['positive'] += 1
            elif neg > pos and neg > neu:
                sentiment_counts['negative'] += 1
            else: # Default to neutral if scores are equal or only neutral is highest
                sentiment_counts['neutral'] += 1
        # else: news item lacks sentiment scores, don't count it for the pie chart

    sentiment_percentages = {}
    if total_analyzed > 0:
        sentiment_percentages['positive'] = round((sentiment_counts['positive'] / total_analyzed) * 100, 1)
        sentiment_percentages['negative'] = round((sentiment_counts['negative'] / total_analyzed) * 100, 1)
        sentiment_percentages['neutral'] = round((sentiment_counts['neutral'] / total_analyzed) * 100, 1)
        # Adjust rounding to ensure total is 100% if needed (e.g., assign remainder to largest category)
        # For simplicity, we'll leave it as is for now.
    else:
        sentiment_percentages = {'positive': 0, 'negative': 0, 'neutral': 100} # Default if no analyzed news

    # --- Fetch Model Predictions ---
    model_predictions = get_model_predictions(symbol)

    # --- Prepare Context ---
    
    # Keys to exclude from display (sensitive, irrelevant, or shown elsewhere)
    excluded_keys = ['longBusinessSummary', 'companyOfficers']
    
    # Keys to prioritize for display (most important first)
    important_keys = [
        'sector', 'industry', 'website', 'marketCap', 'currentPrice', 
        'regularMarketOpen', 'regularMarketDayHigh', 'regularMarketDayLow',
        'regularMarketVolume', 'regularMarketPreviousClose',
        'fiftyDayAverage', 'twoHundredDayAverage',
        'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 
        'dividendRate', 'dividendYield', 'exDividendDate',
        'trailingPE', 'forwardPE', 'priceToSalesTrailing12Months',
                    'bookValue', 'priceToBook', 'earningsGrowth', 'revenueGrowth',
    ]

    # Keys needed for AI analysis in JavaScript - expanded to include more financial metrics
    # Start with the important keys already defined above and add additional metrics
    key_metrics_map_keys = important_keys + [
        'peRatio', 'beta', 'profitMargins', 'operatingMargins', 'grossMargins',
        'returnOnAssets', 'returnOnEquity', 'revenue', 'revenuePerShare',
        'ebitda', 'trailingEps', 'forwardEps', 'totalCash', 'totalDebt',
        'totalRevenue', 'debtToEquity', 'freeCashflow', 'operatingCashflow'
    ]
    # Remove duplicates while preserving order
    key_metrics_map_keys = list(dict.fromkeys(key_metrics_map_keys))
    key_metrics_data = {k: stock_info.get(k) for k in key_metrics_map_keys if stock_info.get(k) is not None}
    
    # Special formatting keys
    epoch_keys = ['exDividendDate', 'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter', 'lastDividendDate']
    large_number_keys = ['marketCap', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 
                        'sharesOutstanding', 'floatShares', 'enterpriseValue', 'totalAssets', 'totalCash',
                        'totalDebt', 'totalRevenue', 'revenue', 'grossProfits', 'freeCashflow', 'operatingCashflow']
    percentage_keys = ['dividendYield', 'profitMargins', 'grossMargins', 'operatingMargins', 'earningsGrowth', 'revenueGrowth',
                      'heldPercentInsiders', 'heldPercentInstitutions']
    currency_keys = ['currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice',
                    'regularMarketOpen', 'regularMarketDayHigh', 'regularMarketDayLow', 'regularMarketPrice',
                    'regularMarketPreviousClose', 'ask', 'bid', 'dividendRate', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow',
                    'fiftyDayAverage', 'twoHundredDayAverage', 'trailingPE', 'forwardPE', 'priceToBook',
                    'bookValue', 'enterpriseToRevenue', 'enterpriseToEbitda']

    context = {
        # 'stock': stock, # REMOVED - This was causing the error
        'stock_data': stock_data, # Pass the whole dictionary from MongoDB
        'stock_info': stock_info, # Pass the extracted info dictionary for convenience (optional)
        'stock_historical_data': stock_historical_data, # Use the fetched historical data
        'chart_labels': json.dumps(chart_dates),
        'chart_close': json.dumps(chart_close),
        'chart_open': json.dumps(chart_open),
        'chart_high': json.dumps(chart_high),
        'chart_low': json.dumps(chart_low),
        'chart_volume': json.dumps(chart_volume),
        'commodity_chart_data': commodity_chart_data, # Add commodity data
        'excluded_keys': excluded_keys,
        'important_keys': important_keys,
        'epoch_keys': epoch_keys,
        'large_number_keys': large_number_keys,
        'percentage_keys': percentage_keys,
        'currency_keys': currency_keys,
        'news_items': news_items, # Add news items to context
        'sentiment_percentages': json.dumps(sentiment_percentages), # Add sentiment data for chart
        'sentiment_counts': sentiment_counts, # Optional: pass counts too
        'total_analyzed_news': total_analyzed, # Optional: pass total analyzed count
        'model_predictions': model_predictions, # Add model predictions to context
        'key_metrics_json': json.dumps(key_metrics_data), # Add key metrics for JS AI analysis
    }

    return render(request, 'stocks/stock_detail.html', context)

# Note: This view implicitly uses POST, consider adding @require_POST decorator
def update_stock_data(request: HttpRequest, symbol: str):
    """Update stock data from API."""
    # Ensure it's a POST request for security (avoid CSRF issues)
    if request.method == 'POST':
        symbol = symbol.upper()
        stock = get_stock_data(symbol, force_update=True)
        
        if stock:
            messages.success(request, f"Data for {symbol} has been updated.")
        else:
            messages.error(request, f"Failed to update data for {symbol}.")
            
        # Redirect back to stock detail page
        return redirect('stock_detail', symbol=symbol)
    
    # If not POST, redirect to home
    return redirect('home')

@login_required
def watchlist(request: HttpRequest):
    """Display the user's watchlist."""
    # Get or create the user's watchlist
    watchlist, _created = Watchlist.objects.get_or_create(user=request.user) # Prefix unused 'created'

    # Get all watchlist items
    items = watchlist.items.all()
    
    # Prepare data for template
    purchased_items = []
    tracked_items = []
    total_investment = Decimal('0.00')
    total_current_value = Decimal('0.00')
    
    # Fetch current prices and prepare data
    for item in items:
        # Update the price if it's been more than a day since last check or if it's not set
        should_update = (not item.last_price_check or 
                       (datetime.now().replace(tzinfo=None) - item.last_price_check.replace(tzinfo=None) > timedelta(days=1)))
        
        if should_update:
            try:
                ticker = yf.Ticker(item.symbol)
                # Get the last price and basic info
                today_data = ticker.history(period='1d')
                if not today_data.empty:
                    item.last_price = today_data['Close'][-1]
                    item.last_price_check = datetime.now()
                    item.save()
            except Exception as e:
                print(f"Error updating price for {item.symbol}: {e}")
        
        # Get company name and daily change
        try:
            ticker = yf.Ticker(item.symbol)
            info = ticker.info
            name = info.get('shortName', info.get('longName', item.symbol))
            
            # Calculate daily change
            today_data = ticker.history(period='2d')
            if len(today_data) >= 2:
                yesterday_close = today_data['Close'][-2]
                today_close = today_data['Close'][-1]
                daily_change = today_close - yesterday_close
                daily_change_percent = (daily_change / yesterday_close) * 100 if yesterday_close > 0 else 0
            else:
                daily_change = None
                daily_change_percent = None
                
            item_detail = {
                'item': item,
                'name': name,
                'current_price': item.last_price,
                'daily_change': daily_change,
                'daily_change_percent': daily_change_percent
            }
            
            # Add to appropriate list and calculate totals
            if item.tracking_type == 'PURCHASED':
                purchased_items.append(item_detail)
                if item.total_investment:
                    total_investment += item.total_investment
                if item.current_value:
                    total_current_value += item.current_value
            else:
                tracked_items.append(item_detail)
        
        except Exception as e:
            print(f"Error processing {item.symbol}: {e}")
            # Still add the item but with limited info
            item_detail = {
                'item': item,
                'name': item.symbol,
                'current_price': item.last_price,
                'daily_change': None,
                'daily_change_percent': None
            }
            if item.tracking_type == 'PURCHASED':
                purchased_items.append(item_detail)
            else:
                tracked_items.append(item_detail)
    
    context = {
        'purchased_items': purchased_items,
        'tracked_items': tracked_items,
        'total_investment': total_investment,
        'total_current_value': total_current_value,
    }
    
    return render(request, 'stocks/watchlist.html', context)

@login_required
def add_to_watchlist(request: HttpRequest, symbol: str = None):
    """Add a stock to the watchlist."""
    # Get or create the user's watchlist
    watchlist, _created = Watchlist.objects.get_or_create(user=request.user) # Prefix unused 'created'

    # If a symbol is provided in the URL, pre-fill the form with it
    initial_data = {}
    if symbol:
        initial_data['symbol'] = symbol.upper()
        
    if request.method == 'POST':
        form = WatchlistItemForm(request.POST)
        if form.is_valid():
            try:
                # Create but don't save the new watchlist item yet
                watchlist_item = form.save(commit=False)
                watchlist_item.watchlist = watchlist
                
                # Uppercase the symbol for consistency
                watchlist_item.symbol = watchlist_item.symbol.upper()
                
                # Try to get current price
                try:
                    ticker = yf.Ticker(watchlist_item.symbol)
                    today_data = ticker.history(period='1d')
                    if not today_data.empty:
                        watchlist_item.last_price = today_data['Close'][-1]
                        watchlist_item.last_price_check = datetime.now()
                except Exception as e:
                    print(f"Error getting current price: {e}")
                
                # Save the watchlist item
                watchlist_item.save()
                messages.success(request, f"{watchlist_item.symbol} added to your watchlist.")
                return redirect('watchlist')
                
            except IntegrityError:
                # Handle the case where the stock already exists in the watchlist
                messages.error(request, f"{form.cleaned_data['symbol']} is already in your watchlist.")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = WatchlistItemForm(initial=initial_data)
    
    context = {
        'form': form,
        'symbol': symbol,
    }
    
    return render(request, 'stocks/add_to_watchlist.html', context)

@login_required
def add_to_watchlist_with_symbol(request: HttpRequest, symbol: str):
    """
    View that pre-fills the symbol when adding a stock to watchlist.
    It's a convenience wrapper around add_to_watchlist.
    """
    return add_to_watchlist(request, symbol)

@login_required
def edit_watchlist_item(request: HttpRequest, item_id: int):
    """Edit a watchlist item."""
    # Get the watchlist item
    watchlist_item = get_object_or_404(WatchlistItem, id=item_id, watchlist__user=request.user)
    
    if request.method == 'POST':
        form = WatchlistItemForm(request.POST, instance=watchlist_item)
        if form.is_valid():
            form.save()
            messages.success(request, f"{watchlist_item.symbol} has been updated.")
            return redirect('watchlist')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = WatchlistItemForm(instance=watchlist_item)
    
    context = {
        'form': form,
        'item': watchlist_item,
    }
    
    return render(request, 'stocks/edit_watchlist_item.html', context)

# Note: This view implicitly uses POST, consider adding @require_POST decorator
@login_required
def remove_from_watchlist(request: HttpRequest, item_id: int):
    """Remove a stock from the watchlist."""
    # Ensure it's a POST request for security
    if request.method == 'POST':
        # Get the watchlist item
        watchlist_item = get_object_or_404(WatchlistItem, id=item_id, watchlist__user=request.user)
        symbol = watchlist_item.symbol
        watchlist_item.delete()
        messages.success(request, f"{symbol} has been removed from your watchlist.")
    
    return redirect('watchlist')


@login_required
def account_view(request: HttpRequest):
    """Display and handle updates for user account information and password."""
    if request.method == 'POST':
        # Check which form was submitted based on button name or hidden field
        if 'update_profile' in request.POST:
            user_form = UserUpdateForm(request.POST, instance=request.user)
            password_form = PasswordChangeForm(request.user) # Keep password form unbound
            if user_form.is_valid():
                user_form.save()
                messages.success(request, 'Your profile has been updated successfully.')
                return redirect('account') # Redirect back to account page
            else:
                 messages.error(request, 'Please correct the errors below in the profile form.')
        elif 'change_password' in request.POST:
            user_form = UserUpdateForm(instance=request.user) # Keep user form unbound
            password_form = PasswordChangeForm(request.user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                # Important: Update the session auth hash to keep the user logged in
                update_session_auth_hash(request, user) 
                messages.success(request, 'Your password was successfully updated.')
                return redirect('account') # Redirect back to account page
            else:
                messages.error(request, 'Please correct the errors below in the password change form.')
        else:
             # Handle unexpected POST data if necessary
             user_form = UserUpdateForm(instance=request.user)
             password_form = PasswordChangeForm(request.user)
             messages.warning(request, 'Invalid form submission.')

    else: # GET request
        user_form = UserUpdateForm(instance=request.user)
        password_form = PasswordChangeForm(request.user)

    context = {
        'user_form': user_form,
        'password_form': password_form
    }
    return render(request, 'stocks/account.html', context)

def logout_view(request: HttpRequest):
    """
    Custom logout view that ensures proper session cleanup.
    """
    # Explicitly flush the session
    request.session.flush()
    
    # Django's built-in logout function
    logout(request)
    
    # Add a success message
    messages.success(request, "You have been successfully logged out.")
    
    # Redirect to home page
    return redirect('home')

@login_required
def stock_news(request: HttpRequest, symbol: str = None):
    """
    View to fetch and display news for a specific stock symbol with optional date filtering.
    Shows prefetched news from the database with sentiment scores.
    """
    # Default to the last 7 days if no dates provided
    end_date = request.GET.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    # Default start date is 7 days before end date
    try:
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        default_start_date = (end_date_obj - timedelta(days=7)).strftime('%Y-%m-%d')
    except ValueError:
        end_date = datetime.now().strftime('%Y-%m-%d')
        default_start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    start_date = request.GET.get('start_date', default_start_date)
    
    # If symbol is provided in URL, use it; otherwise check for GET parameter
    if not symbol:
        symbol = request.GET.get('symbol', '')
    
    symbol = symbol.upper() if symbol else ''
    
    # Convert string dates to datetime objects for filtering
    try:
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        # Add a day to end_date to include the full end date
        end_date_obj = end_date_obj + timedelta(days=1)
    except ValueError:
        start_date_obj = datetime.now() - timedelta(days=7)
        end_date_obj = datetime.now() + timedelta(days=1)
        start_date = start_date_obj.strftime('%Y-%m-%d')
        end_date = end_date_obj.strftime('%Y-%m-%d')
    
    error_message = None
    
    # Query for news articles from the database
    if symbol:
        try:
            # Get stock object or 404
            try:
                stock_obj = Stock.objects.get(symbol=symbol)
                
                # Query StockNews model for the specific stock and date range
                news_items = StockNews.objects.filter(
                    stock=stock_obj,
                    published_at__gte=start_date_obj,
                    published_at__lt=end_date_obj
                ).order_by('-published_at')
                
                if not news_items.exists():
                    error_message = f"No news articles found for {symbol} in the selected date range."
                
            except Stock.DoesNotExist:
                error_message = f"Stock with symbol '{symbol}' does not exist."
                news_items = []
                
        except Exception as e:
            logger.error(f"Error retrieving news for {symbol}: {e}")
            error_message = f"An error occurred while retrieving news: {str(e)}"
            news_items = []
    else:
        # If no symbol specified, get recent news across all stocks
        news_items = StockNews.objects.filter(
            published_at__gte=start_date_obj,
            published_at__lt=end_date_obj
        ).order_by('-published_at')[:50]  # Limit to 50 most recent articles
    
    context = {
        'symbol': symbol,
        'news_items': news_items,
        'start_date': start_date,
        'end_date': end_date,
        'error_message': error_message,
    }
    
    return render(request, 'stocks/stock_news.html', context)

@login_required
def ajax_get_ai_analysis(request: HttpRequest, symbol: str):
    """
    Endpoint to get AI-powered stock analysis
    Uses either Gemini or local LLM based on request parameter
    
    Accepts both:
    - GET requests with analyzer as query parameter
    - POST requests with JSON payload containing stock data collected from the frontend
    """
    if request.method == 'POST':
        try:
            # Parse the JSON data sent from frontend
            data = json.loads(request.body)
            analyzer_choice = data.get('analyzer')
            
            # Call the AI service with the collected data
            analysis_result = get_ai_stock_analysis(symbol, analyzer_choice=analyzer_choice, collected_data=data)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Error processing AI analysis request: {e}")
            return JsonResponse({'error': f"Error processing request: {str(e)}"}, status=500)
    else:
        # Handle GET requests for backward compatibility
        analyzer_choice = request.GET.get('analyzer')
        analysis_result = get_ai_stock_analysis(symbol, analyzer_choice=analyzer_choice)
    
    # Return the result as a JSON response
    return JsonResponse(analysis_result)
