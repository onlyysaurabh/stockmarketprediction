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
import subprocess
import os
import sys
import tempfile
from pathlib import Path

from .models import Stock, StockNews
from .forms import FetchNewsForm, TrainModelForm
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

@staff_member_required
def train_models_view(request: HttpRequest) -> TemplateResponse | HttpResponseRedirect:
    """View for training stock prediction models through the admin interface."""
    # Get all unique sectors and industries for the filters
    sectors = Stock.objects.exclude(sector__isnull=True).exclude(sector='').values_list('sector', flat=True).distinct().order_by('sector')
    industries = Stock.objects.exclude(industry__isnull=True).exclude(industry='').values_list('industry', flat=True).distinct().order_by('industry')
    
    # Get all stocks for selection
    all_stocks = Stock.objects.all().order_by('symbol')
    
    # No pre-selected IDs for this view
    preselected_ids = []
    
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
        if form.is_valid():
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            selected_models = form.cleaned_data['models']
            max_workers = form.cleaned_data['max_workers']
            
            # Get selected stocks from the checkboxes
            selected_stock_ids = request.POST.getlist('selected_stocks')
            
            if not selected_stock_ids:
                messages.warning(request, "No stocks selected. Please select at least one stock.")
                context = {
                    **admin.site.each_context(request),
                    'title': 'Train Stock Prediction Models',
                    'form': form,
                    'all_stocks': all_stocks,
                    'sectors': sectors,
                    'industries': industries,
                    'preselected_ids': [],
                    'media': form.media,
                    'opts': Stock._meta,
                }
                return TemplateResponse(request, "admin/stocks/stock/train_models_form.html", context)
            
            # Get the selected stocks - ensure IDs are integers
            selected_stocks = Stock.objects.filter(pk__in=[int(id_val) for id_val in selected_stock_ids])
            selected_symbols = list(selected_stocks.values_list('symbol', flat=True))
            stock_count = len(selected_symbols)
            
            # Create a temporary JSON configuration file for the training job
            with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as temp_config:
                config_path = temp_config.name
                
                # Prepare jobs for the config file
                jobs = []
                for model_type in selected_models:
                    # Create job entry for each model with all selected stocks
                    jobs.append({
                        "model": model_type,
                        "symbols": selected_symbols
                    })
                
                # Create the complete config
                config = {
                    "max_workers": max_workers,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "jobs": jobs
                }
                
                # Write JSON to temp file - ensure it's actually written
                json.dump(config, temp_config, indent=2)
                temp_config.flush()
                os.fsync(temp_config.fileno())
                logger.info(f"Created training configuration at: {config_path}")
                logger.info(f"Configuration: {json.dumps(config, indent=2)}")
            
            # Construct the path to the train.py script
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            train_script = base_dir / 'train-model' / 'train.py'
            
            # Check if the script exists
            if not train_script.exists():
                os.unlink(config_path)  # Clean up temp file
                messages.error(request, f"Training script not found: {train_script}")
                logger.error(f"Training script not found at: {train_script}")
                return redirect('admin_train_models')
            
            # Start the training process
            try:
                logger.info(f"Starting training process with script: {train_script}")
                
                # Build command with explicit configuration
                cmd = [sys.executable, str(train_script), '--config', config_path]
                logger.info(f"Running command: {' '.join(cmd)}")
                
                # Use subprocess.Popen to run it in the background
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=base_dir  # Set working directory to ensure proper path resolution
                )
                
                # Add to admin log
                content_type_id = ContentType.objects.get_for_model(Stock).pk
                model_names = [dict(form.fields['models'].choices)[m] for m in selected_models]
                
                LogEntry.objects.log_action(
                    user_id=request.user.id,
                    content_type_id=content_type_id,
                    object_id=None,
                    object_repr=f"Model Training: {', '.join(model_names)} for {len(selected_symbols)} stocks",
                    action_flag=ADDITION,
                    change_message=json.dumps([{
                        "added": {
                            "name": "model training job", 
                            "object": f"{', '.join(selected_symbols[:5])}{'...' if len(selected_symbols) > 5 else ''}"
                        }
                    }])
                )
                
                # Success message
                messages.success(
                    request, 
                    f"Started training {len(selected_models)} model types for {stock_count} stocks. "
                    f"This process will run in the background and may take a while to complete. "
                    f"You can check train_jobs.log for progress."
                )
                
            except Exception as e:
                os.unlink(config_path)  # Clean up temp file
                error_message = f"Error starting training process: {str(e)}"
                messages.error(request, error_message)
                logger.error(error_message)
            
            # Redirect back to the train models view to stay on the same page
            return redirect('admin_train_models')
        else:
            # Form is invalid, show errors
            messages.error(request, "Please correct the errors below.")
    else:
        # Default date range for training
        today = date.today()
        three_years_ago = today.replace(year=today.year - 3)
        form = TrainModelForm(initial={'start_date': three_years_ago, 'end_date': today})
    
    # Try to get stats about previously trained models by checking log
    model_stats = {
        'total_trained': 0
    }
    
    try:
        # Check if train_jobs.log exists and try to count successful jobs
        log_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'train_jobs.log'
        if log_path.exists():
            with open(log_path, 'r') as log_file:
                log_content = log_file.read()
                # Count successful training jobs
                successful_count = log_content.count("status': 'success")
                if successful_count > 0:
                    model_stats['total_trained'] = successful_count
    except Exception as e:
        logger.error(f"Error getting training stats: {str(e)}")
    
    context = {
        **admin.site.each_context(request),
        'title': 'Train Stock Prediction Models',
        'subtitle': 'Select stocks, date range, and models to train',
        'form': form,
        'all_stocks': all_stocks,
        'sectors': sectors,
        'industries': industries,
        'preselected_ids': [],
        'is_standalone': True,
        'model_stats': model_stats,
        'media': form.media,
        'opts': Stock._meta,
    }
    
    # Use template
    return TemplateResponse(request, "admin/stocks/stock/train_models_form.html", context)
