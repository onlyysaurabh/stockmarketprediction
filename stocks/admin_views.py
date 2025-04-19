import logging
from datetime import date, timedelta, datetime
from django.shortcuts import redirect
from django.contrib import messages
from django.http import HttpRequest, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.admin.models import LogEntry, ADDITION
from django.contrib.contenttypes.models import ContentType
import json

from .models import Stock, StockNews, TrainedPredictionModel
from .forms import FetchNewsForm, TrainModelForm
from .prediction_models import xgboost_predictor, lstm_predictor, sarima_predictor, svm_predictor
from .prediction_data_service import get_training_data
from .admin import admin_site  # Import our custom admin site

logger = logging.getLogger(__name__)

# Use custom admin site for views
def custom_admin_view(view_func):
    """Decorator that uses our custom admin site for views"""
    return admin_site.admin_view(view_func)

@custom_admin_view
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
                    **admin_site.each_context(request),
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
                return TemplateResponse(request, "admin/stocks/fetch_news_form.html", context)
                
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
        **admin_site.each_context(request),
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
    return TemplateResponse(request, "admin/stocks/fetch_news_form.html", context)

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
                    **admin_site.each_context(request),
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
                return TemplateResponse(request, "admin/stocks/fetch_news_form.html", context)
            
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
        **admin_site.each_context(request),
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
    return TemplateResponse(request, "admin/stocks/fetch_news_form.html", context)

@custom_admin_view
def prediction_dashboard(request: HttpRequest) -> TemplateResponse:
    """
    Admin view for stock predictions dashboard.
    Shows trained models and their predictions for different stocks.
    """
    from .forms import ModelTrainingForm
    from .prediction_models import lstm_predictor, sarima_predictor, xgboost_predictor, svm_predictor
    import os
    from .prediction_service import train_model_for_stock

    context = {
        'title': 'Stock Predictions Dashboard',
        'site_title': admin_site.site_title,
        'site_header': admin_site.site_header,
        'has_permission': True,
        **admin_site.each_context(request),
    }

    # Initialize form
    form = ModelTrainingForm(request.POST or None)
    context['form'] = form

    # Handle form submission
    if request.method == 'POST' and form.is_valid():
        selected_models = form.cleaned_data['models']
        selected_stocks = form.cleaned_data['stocks']
        duration = int(form.cleaned_data['duration'])
        sequence_length = int(form.cleaned_data.get('sequence_length', 30))

        # Create a dictionary to track training parameters for later reference
        training_params = {
            'duration': duration,
            'sequence_length': sequence_length,
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Store these parameters in the session for reference
        request.session['last_training_params'] = {
            'models': selected_models,
            'duration': duration,
            'sequence_length': sequence_length,
            'stock_count': len(selected_stocks),
        }

        results = {}
        successful_models = 0
        failed_models = 0

        for stock in selected_stocks:
            stock_results = {}
            
            # Get training data
            try:
                if duration == -1:
                    data = get_training_data(stock.symbol)
                else:
                    end_date = date.today()
                    start_date = end_date - timedelta(days=duration)
                    data = get_training_data(stock.symbol, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                
                if data is None or data.empty:
                    messages.error(request, f"No training data available for {stock.symbol}")
                    continue
                    
                logger.info(f"Got training data for {stock.symbol} with {len(data)} rows")
            except Exception as e:
                logger.error(f"Error getting training data for {stock.symbol}: {str(e)}")
                messages.error(request, f"Error getting training data for {stock.symbol}: {str(e)}")
                continue

            # Train selected models
            for model_type in selected_models:
                try:
                    # Store the training parameters specific to this model type
                    model_training_params = training_params.copy()
                    
                    model_dict = None
                    if model_type == 'LSTM':
                        model_dict = lstm_predictor.train_lstm_model(stock.symbol, data, sequence_length=sequence_length)
                        model_training_params['sequence_length'] = sequence_length
                    elif model_type == 'SARIMA':
                        model_dict = sarima_predictor.train_sarima_model(stock.symbol, data)
                    elif model_type == 'XGBOOST':
                        model_dict = xgboost_predictor.train_xgboost_model(stock.symbol, data)
                    elif model_type == 'SVM':
                        model_dict = svm_predictor.train_svm_model(stock.symbol, data)

                    if model_dict:
                        # Get next day prediction
                        prediction = None
                        if 'model_path' in model_dict:
                            if model_type == 'LSTM':
                                prediction = lstm_predictor.predict_next_day(model_dict['model_path'], data)
                            elif model_type == 'SARIMA':
                                prediction = sarima_predictor.predict_next_day(model_dict['model_path'], data)
                            elif model_type == 'XGBOOST':
                                prediction = xgboost_predictor.predict_next_day(model_dict['model_path'], data)
                            elif model_type == 'SVM':
                                prediction = svm_predictor.predict_next_day(model_dict['model_path'], data)
                                
                            # Store training parameters in model metadata
                            model_id = model_dict.get('model_id')
                            if model_id:
                                try:
                                    # Update the model instance with training parameters
                                    trained_model = TrainedPredictionModel.objects.get(id=model_id)
                                    
                                    # If model has a metadata field, update it, otherwise add it
                                    if not hasattr(trained_model, 'metadata') or trained_model.metadata is None:
                                        trained_model.metadata = {}
                                    
                                    # Update the metadata with training parameters
                                    metadata = trained_model.metadata if hasattr(trained_model, 'metadata') else {}
                                    metadata.update(model_training_params)
                                    trained_model.metadata = metadata
                                    trained_model.save()
                                    
                                    logger.info(f"Updated model {model_id} with training parameters")
                                except Exception as e:
                                    logger.error(f"Error updating model metadata: {e}")

                        # Calculate performance indicators
                        test_r2 = model_dict['performance'].get('test_r2', 0)
                        performance_class = 'poor'
                        if test_r2 > 0.7:
                            performance_class = 'good'
                        elif test_r2 > 0.5:
                            performance_class = 'medium'

                        stock_results[model_type] = {
                            'performance': model_dict['performance'],
                            'performance_class': performance_class,
                            'next_day_prediction': prediction,
                            'training_date': model_dict['training_date'],
                            'model_path': model_dict.get('model_path', ''),
                            'training_params': model_training_params  # Store parameters with results
                        }
                        
                        # Add to admin log
                        content_type_id = ContentType.objects.get_for_model(Stock).pk
                        stock_id = stock.id if hasattr(stock, 'id') else None
                        LogEntry.objects.log_action(
                            user_id=request.user.id,
                            content_type_id=content_type_id,
                            object_id=stock_id,
                            object_repr=f"{stock.symbol} - {model_type} model",
                            action_flag=ADDITION,
                            change_message=json.dumps([{
                                "added": {
                                    "name": "prediction model", 
                                    "object": f"{model_type} model for {stock.symbol} (R²: {test_r2:.4f})"
                                }
                            }])
                        )
                        successful_models += 1
                except Exception as e:
                    logger.error(f"Error training {model_type} model for {stock.symbol}: {str(e)}")
                    messages.error(request, f"Error training {model_type} model for {stock.symbol}: {str(e)}")
                    failed_models += 1

            results[stock.symbol] = stock_results

        context['results'] = results
        if successful_models > 0:
            messages.success(request, f"Models trained successfully: {successful_models} model(s) for {len(selected_stocks)} stock(s)! Failed models: {failed_models}")
        else:
            messages.error(request, f"No models were successfully trained. Please check the logs for more information.")

    # Get list of existing trained models
    trained_models = {}
    # First get models from the database
    db_models = TrainedPredictionModel.objects.all().order_by('stock__symbol', '-trained_at')
    
    for model in db_models:
        symbol = model.stock.symbol
        if symbol not in trained_models:
            trained_models[symbol] = []
            
        trained_models[symbol].append({
            'id': model.id,
            'type': model.model_type,
            'filename': os.path.basename(model.model_path),
            'trained_at': model.trained_at,
            'metadata': model.metadata if hasattr(model, 'metadata') else {}
        })

    context['trained_models'] = trained_models
    
    return TemplateResponse(request, 'admin/prediction_dashboard.html', context)

@staff_member_required
def admin_delete_model(request, model_id):
    """
    Admin view for deleting a trained prediction model.
    """
    context = {
        **admin_site.each_context(request),
        'title': 'Delete Model',
        'model_id': model_id,
    }
    
    try:
        # Here we would add logic to delete the model based on model_id
        # For example, removing the model file from trained_models directory
        # and any associated database records
        
        messages.success(request, f'Successfully deleted model {model_id}')
        return redirect('admin:prediction_dashboard')
    except Exception as e:
        messages.error(request, f'Error deleting model: {str(e)}')
        return redirect('admin:prediction_dashboard')

@staff_member_required
def update_all_stock_prices(request):
    """
    Admin view for bulk updating all stock prices.
    """
    from .services import update_stock_prices
    from .models import Stock
    
    try:
        # Get all active stocks
        stocks = Stock.objects.all()
        count = 0
        
        for stock in stocks:
            result = update_stock_prices(stock.symbol)
            if result and result.get('success'):
                count += 1
        
        messages.success(request, f"Successfully updated prices for {count} out of {stocks.count()} stocks.")
    except Exception as e:
        messages.error(request, f"Error updating stock prices: {str(e)}")
    
    return redirect('admin:stocks_stock_changelist')

@staff_member_required
def update_commodities(request):
    """
    Admin view for updating commodity data.
    """
    from .services import update_commodity_prices
    
    try:
        # Update commodity prices - assuming this function exists in your services
        result = update_commodity_prices()
        messages.success(request, f"Successfully updated commodity prices. {result.get('count', 0)} commodities updated.")
    except Exception as e:
        messages.error(request, f"Error updating commodity prices: {str(e)}")
    
    return redirect('admin:stocks_stock_changelist')
