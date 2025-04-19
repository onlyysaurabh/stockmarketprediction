from django.shortcuts import redirect
from django.contrib import messages
from django.http import HttpRequest, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.admin.models import LogEntry, ADDITION
from django.contrib.contenttypes.models import ContentType
import json
from datetime import date, timedelta, datetime
import logging

from .models import Stock, StockNews, TrainedPredictionModel
from .forms import FetchNewsForm, TrainModelForm
from .prediction_models import xgboost_predictor, lstm_predictor, sarima_predictor, svm_predictor
from .prediction_data_service import get_training_data
from .services import (
    fetch_and_store_stock_data, 
    fetch_and_store_commodity_data,
    process_news_for_stock
)
from .admin_utils import fetch_stock_news_view

logger = logging.getLogger(__name__)

@staff_member_required
def fetch_news_standalone_view(request):
    """
    Standalone view for fetching news for stocks with date range selection
    """
    if request.method == 'POST':
        form = FetchNewsForm(request.POST)
        if form.is_valid():
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            selected_stocks = form.cleaned_data['stocks']
            
            total_news_items = 0
            success_stocks = []
            failed_stocks = []
            
            for stock in selected_stocks:
                try:
                    news_items = process_news_for_stock(stock, start_date, end_date)
                    if news_items:
                        total_news_items += len(news_items)
                        success_stocks.append(stock.symbol)
                        
                        # Log the action
                        LogEntry.objects.log_action(
                            user_id=request.user.id,
                            content_type_id=ContentType.objects.get_for_model(StockNews).pk,
                            object_id=stock.id,
                            object_repr=stock.symbol,
                            action_flag=ADDITION,
                            change_message=json.dumps([{
                                'added': {
                                    'name': 'Stock News',
                                    'object': f'{len(news_items)} news items'
                                }
                            }])
                        )
                except Exception as e:
                    logger.error(f"Error fetching news for {stock.symbol}: {str(e)}")
                    failed_stocks.append(stock.symbol)
            
            if success_stocks:
                messages.success(
                    request,
                    f'Successfully fetched {total_news_items} news items for {", ".join(success_stocks)}'
                )
            
            if failed_stocks:
                messages.error(
                    request,
                    f'Failed to fetch news for the following stocks: {", ".join(failed_stocks)}'
                )
            
            if success_stocks:
                return redirect('admin:stocks_stocknews_changelist')
    else:
        form = FetchNewsForm()
    
    context = {
        'title': 'Fetch Stock News',
        'form': form,
        'is_nav_sidebar_enabled': True,
        'has_permission': True,
    }
    
    return TemplateResponse(request, 'admin/fetch_news_standalone.html', context)

@staff_member_required
def update_commodities(request):
    """
    Admin view to update commodity prices.
    Common commodity futures: GC=F (Gold), SI=F (Silver), CL=F (Crude Oil), etc.
    """
    commodity_symbols = ['GC=F', 'SI=F', 'CL=F', 'HG=F', 'NG=F']  # Gold, Silver, Crude Oil, Copper, Natural Gas
    updated_count = 0
    
    try:
        for symbol in commodity_symbols:
            try:
                # Force update for each commodity
                fetch_and_store_commodity_data(symbol, force_update=True)
                updated_count += 1
            except Exception as e:
                logger.error(f"Error updating commodity {symbol}: {str(e)}")
                messages.error(request, f"Failed to update {symbol}: {str(e)}")
        
        if updated_count > 0:
            messages.success(request, f"Successfully updated prices for {updated_count} commodities.")
        else:
            messages.warning(request, "No commodities were updated.")
            
    except Exception as e:
        logger.error(f"Error in update_commodities: {str(e)}")
        messages.error(request, f"An error occurred while updating commodity prices: {str(e)}")
    
    # Redirect back to the stock list in admin
    return redirect('admin:stocks_stock_changelist')

@staff_member_required
def update_all_stock_prices(request):
    """
    Admin view to update prices for all active stocks.
    """
    try:
        updated_count = 0
        stocks = Stock.objects.filter(is_active=True)
        
        for stock in stocks:
            try:
                # Force update for each stock
                fetch_and_store_stock_data(stock.symbol, force_update=True)
                updated_count += 1
            except Exception as e:
                logger.error(f"Error updating stock {stock.symbol}: {str(e)}")
                messages.error(request, f"Failed to update {stock.symbol}: {str(e)}")
        
        if updated_count > 0:
            messages.success(request, f"Successfully updated prices for {updated_count} stocks.")
        else:
            messages.warning(request, "No stocks were updated.")
            
    except Exception as e:
        logger.error(f"Error in update_all_stock_prices: {str(e)}")
        messages.error(request, f"An error occurred while updating stock prices: {str(e)}")
    
    # Redirect back to the stock list in admin
    return redirect('admin:stocks_stock_changelist')

@staff_member_required
def prediction_dashboard(request):
    """
    Admin dashboard for managing stock predictions.
    """
    # Get all stocks with active predictions
    stocks = Stock.objects.filter(is_active=True)
    trained_models = TrainedPredictionModel.objects.all().order_by('-trained_at')
    
    context = {
        'title': 'Prediction Dashboard',
        'stocks': stocks,
        'trained_models': trained_models,
        'available_models': ['XGBoost', 'LSTM', 'SARIMA', 'SVM'],
    }
    
    return TemplateResponse(request, 'admin/prediction_dashboard.html', context)
