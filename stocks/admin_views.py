import logging
from datetime import date, timedelta
from django.contrib import admin
from django.shortcuts import redirect
from django.contrib import messages
from django.http import HttpRequest, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.admin.models import LogEntry, ADDITION
from django.contrib.contenttypes.models import ContentType
import json

from .models import Stock, StockNews
from .forms import FetchNewsForm
from .services import process_news_for_stock

logger = logging.getLogger(__name__)

@admin.site.admin_view
def fetch_stock_news_view(request: HttpRequest) -> TemplateResponse | HttpResponseRedirect:
    """
    Custom admin view to fetch news for selected stocks within a date range.
    Allows filtering stocks by sector and industry.
    """
    # Get all unique sectors and industries for the filters
    sectors = Stock.objects.exclude(sector__isnull=True).exclude(sector='').values_list('sector', flat=True).distinct().order_by('sector')
    industries = Stock.objects.exclude(industry__isnull=True).exclude(industry='').values_list('industry', flat=True).distinct().order_by('industry')
    
    # Get all stocks for selection
    all_stocks = Stock.objects.all().order_by('symbol')
    
    # Get pre-selected stock IDs from query parameters
    preselected_ids = []
    selected_ids_str = request.GET.get('ids')
    if selected_ids_str:
        try:
            # Convert string IDs to integers for proper database querying
            preselected_ids = [int(id_str) for id_str in selected_ids_str.split(',')]
            selected_stocks = Stock.objects.filter(pk__in=preselected_ids)
            stock_symbols = list(selected_stocks.values_list('symbol', flat=True))
            # Convert back to strings for template comparison
            preselected_ids = [str(id_val) for id_val in preselected_ids]
        except (ValueError, TypeError):
            messages.error(request, "Invalid selection provided.")
            return redirect('admin:stocks_stock_changelist')
    else:
        selected_stocks = Stock.objects.none()
        stock_symbols = []

    if request.method == 'POST':
        form = FetchNewsForm(request.POST)
        if form.is_valid():
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            
            # Get selected stocks from the checkboxes
            selected_stock_ids = request.POST.getlist('selected_stocks')
            
            if not selected_stock_ids:
                messages.warning(request, "No stocks selected. Please select at least one stock.")
                context = {
                    **admin.site.each_context(request),
                    'title': 'Fetch News for Stocks',
                    'form': form,
                    'selected_stocks': selected_stocks,
                    'all_stocks': all_stocks,
                    'sectors': sectors,
                    'industries': industries,
                    'preselected_ids': preselected_ids,
                    'stock_symbols': stock_symbols,
                    'media': form.media,
                    'opts': Stock._meta,
                }
                return TemplateResponse(request, "admin/stocks/stock/fetch_news_form.html", context)
                
            # Get the selected stocks - ensure IDs are integers
            selected_stocks = Stock.objects.filter(pk__in=[int(id_val) for id_val in selected_stock_ids])
            selected_symbols = list(selected_stocks.values_list('symbol', flat=True))
            stock_count = len(selected_symbols)
            
            # Process the news fetching for each stock
            total_processed = 0
            total_created = 0
            total_updated = 0
            total_failed = 0
            
            for symbol in selected_symbols:
                # Process news for this stock with the given date range
                result = process_news_for_stock(symbol, start_date=start_date, end_date=end_date)
                
                total_processed += result.get('processed', 0)
                total_created += result.get('created', 0)
                total_updated += result.get('updated', 0)
                if result.get('failed', 0) > 0:
                    total_failed += 1  # Count stocks with failures

            messages.success(request, 
                            f"News fetch completed for {stock_count} stocks ({start_date} to {end_date}). "
                            f"Articles: {total_processed} processed, {total_created} created, "
                            f"{total_updated} updated, {total_failed} stocks with failures.")
            
            # Redirect back to the stock changelist after processing
            return redirect('admin:stocks_stock_changelist')
        else:
            # Form is invalid, show errors
            messages.error(request, "Please correct the date errors below.")

    else:  # GET request
        # Default date range (e.g., last 7 days)
        today = date.today()
        seven_days_ago = today - timedelta(days=7)
        form = FetchNewsForm(initial={'start_date': seven_days_ago, 'end_date': today})

    context = {
        **admin.site.each_context(request),
        'title': 'Fetch News for Stocks',
        'form': form,
        'selected_stocks': selected_stocks,
        'all_stocks': all_stocks,
        'sectors': sectors,
        'industries': industries,
        'preselected_ids': preselected_ids,
        'stock_symbols': stock_symbols,
        'media': form.media,
        'opts': Stock._meta,
    }
    
    # Use the enhanced template for this view
    return TemplateResponse(request, "admin/stocks/stock/fetch_news_form.html", context)

@staff_member_required
def fetch_news_standalone_view(request: HttpRequest) -> TemplateResponse | HttpResponseRedirect:
    """Standalone view for fetching news with full stock selection capability"""
    # Get all unique sectors and industries for the filters
    sectors = Stock.objects.exclude(sector__isnull=True).exclude(sector='').values_list('sector', flat=True).distinct().order_by('sector')
    industries = Stock.objects.exclude(industry__isnull=True).exclude(industry='').values_list('industry', flat=True).distinct().order_by('industry')
    
    # Get all stocks for selection
    all_stocks = Stock.objects.all().order_by('symbol')
    
    # Initialize pre-selected stocks (none by default for the standalone view)
    preselected_ids = []
    
    if request.method == 'POST':
        form = FetchNewsForm(request.POST)
        if form.is_valid():
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            
            # Get selected stocks from the checkboxes
            selected_stock_ids = request.POST.getlist('selected_stocks')
            
            if not selected_stock_ids:
                messages.warning(request, "No stocks selected. Please select at least one stock.")
                context = {
                    **admin.site.each_context(request),
                    'title': 'Fetch Stock News',
                    'form': form,
                    'all_stocks': all_stocks,
                    'sectors': sectors,
                    'industries': industries,
                    'preselected_ids': [],
                    'is_standalone': True,
                    'media': form.media,
                    'opts': Stock._meta,
                }
                return TemplateResponse(request, "admin/stocks/stock/fetch_news_form.html", context)
            
            # Get the selected stocks - ensure IDs are integers
            selected_stocks = Stock.objects.filter(pk__in=[int(id_val) for id_val in selected_stock_ids])
            selected_symbols = list(selected_stocks.values_list('symbol', flat=True))
            stock_count = len(selected_symbols)
            
            # Process the news fetching for each stock
            total_processed = 0
            total_created = 0 
            total_updated = 0
            total_failed = 0
            
            for symbol in selected_symbols:
                # Process news for this stock with the given date range
                result = process_news_for_stock(symbol, start_date=start_date, end_date=end_date)
                
                total_processed += result.get('processed', 0)
                total_created += result.get('created', 0)
                total_updated += result.get('updated', 0)
                if result.get('failed', 0) > 0:
                    total_failed += 1  # Count stocks with failures

            # Show success message
            messages.success(request, 
                            f"News fetch completed for {stock_count} stocks ({start_date} to {end_date}). "
                            f"Articles: {total_processed} processed, {total_created} created, "
                            f"{total_updated} updated, {total_failed} stocks with failures.")
            
            # Add to admin log for showing in "Recent actions"
            content_type_id = ContentType.objects.get_for_model(StockNews).pk
            display_symbols = selected_symbols[:5]
            LogEntry.objects.log_action(
                user_id=request.user.id,
                content_type_id=content_type_id,
                object_id=None,
                object_repr=f"Standalone News Fetch: {', '.join(display_symbols)}{'...' if len(selected_symbols) > 5 else ''}",
                action_flag=ADDITION,
                change_message=json.dumps([{
                    "added": {
                        "name": "stock news", 
                        "object": f"{total_created} new articles from {start_date} to {end_date}"
                    }
                }])
            )
            
            # Either redirect to StockNews admin or back to this page based on user choice
            if request.POST.get('continue_fetching'):
                return redirect(reverse('admin_fetch_news_standalone'))
            elif request.POST.get('view_news'):
                return redirect('admin:stocks_stocknews_changelist')
            else:
                return redirect('admin:stocks_stock_changelist')
        else:
            # Form is invalid, show errors
            messages.error(request, "Please correct the date errors below.")
    else:
        # Default date range (e.g., last 7 days)
        today = date.today()
        seven_days_ago = today - timedelta(days=7)
        form = FetchNewsForm(initial={'start_date': seven_days_ago, 'end_date': today})
    
    # Get stats about news in the database for display
    news_stats = {
        'total_news': StockNews.objects.count(),
        'recent_news': StockNews.objects.filter(published_at__gte=date.today() - timedelta(days=7)).count(),
        'stock_count': Stock.objects.filter(news__isnull=False).distinct().count(),
    }
    
    context = {
        **admin.site.each_context(request),
        'title': 'Fetch Stock News',
        'subtitle': 'Select stocks and date range to fetch news articles',
        'form': form,
        'all_stocks': all_stocks,
        'sectors': sectors,
        'industries': industries,
        'preselected_ids': [],
        'is_standalone': True,
        'news_stats': news_stats,
        'media': form.media,
        'opts': Stock._meta,
    }
    
    # Use the fetch news template
    return TemplateResponse(request, "admin/stocks/stock/fetch_news_form.html", context)

import logging
import json
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpRequest
from django.contrib import messages
from django.urls import reverse
from django.utils.html import format_html
from django.views.decorators.http import require_POST
from django.db import transaction

from .models import Stock, TrainedPredictionModel
from .prediction_service import train_model_for_stock, get_prediction_for_stock, get_multiple_predictions

logger = logging.getLogger(__name__)

@staff_member_required
def admin_prediction_dashboard(request):
    """
    Admin dashboard for model training and predictions
    """
    stocks = Stock.objects.all().order_by('symbol')
    trained_models = TrainedPredictionModel.objects.all().order_by('-trained_at')
    
    # Get count of models per type
    model_counts = {}
    for model_type, _ in TrainedPredictionModel.MODEL_TYPES:
        model_counts[model_type] = TrainedPredictionModel.objects.filter(model_type=model_type).count()
    
    # Get the most recent model for each stock
    recent_models = {}
    for stock in stocks:
        stock_models = TrainedPredictionModel.objects.filter(stock=stock).order_by('-trained_at')
        if stock_models.exists():
            recent_models[stock.symbol] = stock_models.first()
    
    # Provide model choices for the dropdown
    model_choices = TrainedPredictionModel.MODEL_TYPES
    
    # Default date ranges for quick selection
    today = date.today()
    date_presets = {
        'month': {
            'start_date': today - timedelta(days=30),
            'end_date': today
        },
        'year': {
            'start_date': today - timedelta(days=365),
            'end_date': today
        },
        'five_years': {
            'start_date': today - timedelta(days=365*5),
            'end_date': today
        }
    }
    
    # Include admin site context for proper sidebar rendering
    context = {
        **admin.site.each_context(request),
        'stocks': stocks,
        'trained_models': trained_models[:20],  # limit to 20 for performance
        'model_counts': model_counts,
        'recent_models': recent_models,
        'model_choices': model_choices,
        'date_presets': date_presets,
        'title': 'Prediction Model Management',
        'opts': Stock._meta,  # This helps with admin breadcrumbs
        'app_label': 'stocks',
    }
    
    return render(request, 'admin/stocks/prediction_dashboard.html', context)

@staff_member_required
@require_POST
def admin_train_model(request):
    """
    Admin view to train a prediction model for a stock
    """
    stock_symbol = request.POST.get('stock_symbol')
    model_type = request.POST.get('model_type')
    start_date_str = request.POST.get('start_date')
    end_date_str = request.POST.get('end_date')
    
    if not stock_symbol or not model_type:
        messages.error(request, 'Stock symbol and model type are required.')
        return redirect(reverse('admin_prediction_dashboard'))
    
    try:
        # Parse dates if provided, otherwise use defaults
        start_date = None
        end_date = None
        days = 365  # Default if dates not provided
        
        if start_date_str and end_date_str:
            from datetime import datetime
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            # Calculate days between dates for the API call
            days = (end_date - start_date).days
            if days <= 0:
                messages.error(request, 'End date must be after start date.')
                return redirect(reverse('admin_prediction_dashboard'))
    except ValueError:
        messages.error(request, 'Invalid date format. Please use YYYY-MM-DD.')
        return redirect(reverse('admin_prediction_dashboard'))
    
    # Train the model asynchronously
    # In a production environment, this should be a background task
    # For now, we'll do it synchronously for simplicity
    result = train_model_for_stock(stock_symbol, model_type, days)
    
    if result and result.get('success'):
        messages.success(
            request, 
            f"Successfully trained {model_type} model for {stock_symbol} (ID: {result.get('model_id')})"
        )
    else:
        error = result.get('error') if result else 'Unknown error'
        messages.error(
            request,
            f"Failed to train {model_type} model for {stock_symbol}: {error}"
        )
    
    return redirect(reverse('admin_prediction_dashboard'))

@staff_member_required
def admin_model_detail(request, model_id):
    """
    Admin view to see details of a trained model
    """
    model = get_object_or_404(TrainedPredictionModel, id=model_id)
    
    # Get prediction using this model
    prediction = get_prediction_for_stock(model.stock.symbol, model.model_type)
    
    # Format feature importance as sorted list for display
    feature_importance = []
    if model.feature_importance:
        # If feature_importance is a nested dict (like XGBoost's), flatten it
        if isinstance(model.feature_importance, dict) and 'shap' in model.feature_importance:
            importance_dict = model.feature_importance['shap']
        else:
            importance_dict = model.feature_importance
            
        # Sort by importance value (descending)
        feature_importance = sorted(
            [(k, v) for k, v in importance_dict.items()],
            key=lambda x: x[1],
            reverse=True
        )
    
    context = {
        **admin.site.each_context(request),
        'model': model,
        'prediction': prediction,
        'feature_importance': feature_importance,
        'title': f'Model Detail: {model.stock.symbol} - {model.get_model_type_display()}',
        'opts': Stock._meta,  # This helps with admin breadcrumbs
        'app_label': 'stocks',
    }
    
    return render(request, 'admin/stocks/model_detail.html', context)

@staff_member_required
def admin_stock_predictions(request, stock_id):
    """
    Admin view to see predictions for a specific stock from all model types
    """
    stock = get_object_or_404(Stock, id=stock_id)
    
    # Get predictions from all available models
    multi_predictions = get_multiple_predictions(stock.symbol, days=7)
    
    # Get single-day predictions from each model type for comparison
    predictions = {}
    for model_type, _ in TrainedPredictionModel.MODEL_TYPES:
        pred = get_prediction_for_stock(stock.symbol, model_type)
        if pred:
            predictions[model_type] = pred
    
    context = {
        **admin.site.each_context(request),
        'stock': stock,
        'predictions': predictions,
        'multi_predictions': multi_predictions,
        'title': f'Predictions for {stock.symbol}',
        'opts': Stock._meta,  # This helps with admin breadcrumbs
        'app_label': 'stocks',
    }
    
    return render(request, 'admin/stocks/stock_predictions.html', context)

@staff_member_required
@require_POST
def admin_delete_model(request, model_id):
    """
    Admin view to delete a trained model
    """
    model = get_object_or_404(TrainedPredictionModel, id=model_id)
    stock_symbol = model.stock.symbol
    model_type = model.model_type
    
    try:
        # Delete the model file if possible
        import os
        if os.path.exists(model.model_path):
            os.remove(model.model_path)
        
        # Delete the model database entry
        model.delete()
        
        messages.success(
            request,
            f"Successfully deleted {model_type} model for {stock_symbol}"
        )
    except Exception as e:
        messages.error(
            request,
            f"Error deleting model: {str(e)}"
        )
    
    return redirect(reverse('admin_prediction_dashboard'))
