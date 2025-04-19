def stock_predictions(request, symbol):
    """
    Display and compare stock price predictions from different models.
    """
    # Uppercase the symbol for consistency
    symbol = symbol.upper()
    
    try:
        # Get the stock object
        stock = get_object_or_404(Stock, symbol=symbol)
        
        # Import required modules
        import sys
        import os
        import pandas as pd
        import numpy as np
        import json
        from datetime import datetime, timedelta
        import yfinance as yf
        
        # Add the project root to the Python path if needed
        BASE_DIR = settings.BASE_DIR
        if BASE_DIR not in sys.path:
            sys.path.append(BASE_DIR)
        
        from stocks.models.training_models import TrainedModel
        from stocks.models.model_factory import ModelFactory
        
        # Get available trained models for this stock
        available_models = TrainedModel.objects.filter(
            stock=stock, 
            status='COMPLETED'
        ).order_by('-created_at')
        
        if not available_models.exists():
            messages.warning(request, f"No trained prediction models found for {symbol}. Please train a model first.")
            return redirect('stock_detail', symbol=symbol)
        
        # Get selected models for comparison (from query params or use most recent)
        selected_model_ids = request.GET.getlist('models')
        
        if selected_model_ids:
            selected_models = TrainedModel.objects.filter(
                id__in=selected_model_ids, 
                stock=stock, 
                status='COMPLETED'
            )
        else:
            # Default to the most recent model of each type
            model_types = available_models.values_list('model_type', flat=True).distinct()
            selected_models = []
            for model_type in model_types:
                latest_model = available_models.filter(model_type=model_type).first()
                if latest_model:
                    selected_models.append(latest_model)
            selected_models = TrainedModel.objects.filter(id__in=[m.id for m in selected_models])
        
        # If no models selected or found, use the most recent one
        if not selected_models.exists():
            selected_models = TrainedModel.objects.filter(id=available_models.first().id)
        
        # Get the historical data for the stock
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Get 1 year of data
        
        # Get stock data from yfinance
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            messages.error(request, f"Could not fetch data for {symbol}.")
            return redirect('stock_detail', symbol=symbol)
        
        # Generate predictions for each selected model
        predictions = {}
        metrics = {}
        
        for trained_model in selected_models:
            try:
                # Load the model
                model_instance = ModelFactory.load_model(
                    trained_model.model_type,
                    trained_model.file_path,
                    symbol
                )
                
                # Make predictions
                horizon = trained_model.forecast_horizon
                model_predictions = model_instance.predict(stock_data, horizon=horizon)
                
                # Store predictions
                predictions[str(trained_model.id)] = {
                    'name': trained_model.name,
                    'model_type': trained_model.model_type,
                    'horizon': horizon,
                    'dates': model_predictions.index.strftime('%Y-%m-%d').tolist(),
                    'values': model_predictions['Predicted_Close'].tolist() if 'Predicted_Close' in model_predictions.columns else model_predictions.iloc[:, 0].tolist(),
                    'lower_bound': model_predictions['Lower_CI'].tolist() if 'Lower_CI' in model_predictions.columns else None,
                    'upper_bound': model_predictions['Upper_CI'].tolist() if 'Upper_CI' in model_predictions.columns else None,
                }
                
                # Store metrics
                metrics[str(trained_model.id)] = trained_model.metrics
                
            except Exception as e:
                logger.error(f"Error generating predictions for model {trained_model.id}: {str(e)}")
                messages.error(request, f"Error generating predictions using {trained_model.name}: {str(e)}")
        
        # Prepare historical data for charts
        historical_dates = stock_data.index.strftime('%Y-%m-%d').tolist()
        historical_prices = stock_data['Close'].tolist()
        
        context = {
            'stock': stock,
            'available_models': available_models,
            'selected_models': selected_models,
            'predictions': json.dumps(predictions),
            'metrics': metrics,
            'historical_dates': json.dumps(historical_dates),
            'historical_prices': json.dumps(historical_prices),
        }
        
        return render(request, 'stocks/stock_predictions.html', context)
    
    except Exception as e:
        logger.error(f"Error in stock_predictions view: {str(e)}")
        messages.error(request, f"An error occurred while generating predictions: {str(e)}")
        return redirect('stock_detail', symbol=symbol)

def model_comparison(request):
    """
    Compare performance metrics across different trained models.
    """
    try:
        # Import required modules
        import sys
        import json
        from datetime import datetime, timedelta
        
        # Add the project root to the Python path if needed
        BASE_DIR = settings.BASE_DIR
        if BASE_DIR not in sys.path:
            sys.path.append(BASE_DIR)
        
        from stocks.models.training_models import TrainedModel
        
        # Filter conditions
        stock_id = request.GET.get('stock')
        model_type = request.GET.get('model_type')
        time_period = request.GET.get('period')
        
        # Base queryset of completed models
        queryset = TrainedModel.objects.filter(status='COMPLETED')
        
        # Apply filters
        if stock_id:
            queryset = queryset.filter(stock_id=stock_id)
            
        if model_type:
            queryset = queryset.filter(model_type=model_type)
            
        if time_period:
            # Filter by creation date
            if time_period == '1m':
                date_threshold = datetime.now() - timedelta(days=30)
            elif time_period == '3m':
                date_threshold = datetime.now() - timedelta(days=90)
            elif time_period == '6m':
                date_threshold = datetime.now() - timedelta(days=180)
            elif time_period == '1y':
                date_threshold = datetime.now() - timedelta(days=365)
            else:
                date_threshold = None
                
            if date_threshold:
                queryset = queryset.filter(created_at__gte=date_threshold)
        
        # Get all available stocks and model types for the filter dropdowns
        all_stocks = Stock.objects.all().order_by('symbol')
        all_model_types = TrainedModel.MODEL_TYPES
        
        # Gather metrics for comparison
        models_data = []
        for model in queryset:
            model_data = {
                'id': model.id,
                'name': model.name,
                'stock_symbol': model.stock.symbol,
                'model_type': model.get_model_type_display(),
                'created_at': model.created_at,
                'metrics': model.metrics,
                'parameters': model.parameters,
            }
            models_data.append(model_data)
        
        # Prepare metrics for visualization
        metrics_by_type = {}
        metrics_by_stock = {}
        
        for model in models_data:
            model_type = model['model_type']
            stock = model['stock_symbol']
            
            # Initialize if not exists
            if model_type not in metrics_by_type:
                metrics_by_type[model_type] = {'rmse': [], 'mae': [], 'r2': []}
                
            if stock not in metrics_by_stock:
                metrics_by_stock[stock] = {'rmse': [], 'mae': [], 'r2': []}
            
            # Add metrics if they exist
            metrics = model['metrics']
            if metrics:
                if 'rmse' in metrics:
                    metrics_by_type[model_type]['rmse'].append(metrics['rmse'])
                    metrics_by_stock[stock]['rmse'].append(metrics['rmse'])
                    
                if 'mae' in metrics:
                    metrics_by_type[model_type]['mae'].append(metrics['mae'])
                    metrics_by_stock[stock]['mae'].append(metrics['mae'])
                    
                if 'r2' in metrics:
                    metrics_by_type[model_type]['r2'].append(metrics['r2'])
                    metrics_by_stock[stock]['r2'].append(metrics['r2'])
        
        # Calculate averages
        for model_type, metrics in metrics_by_type.items():
            for metric_name, values in metrics.items():
                if values:
                    metrics[metric_name] = sum(values) / len(values)
                else:
                    metrics[metric_name] = 0
                    
        for stock, metrics in metrics_by_stock.items():
            for metric_name, values in metrics.items():
                if values:
                    metrics[metric_name] = sum(values) / len(values)
                else:
                    metrics[metric_name] = 0
        
        context = {
            'models': models_data,
            'all_stocks': all_stocks,
            'all_model_types': all_model_types,
            'selected_stock': stock_id,
            'selected_model_type': model_type,
            'selected_period': time_period,
            'metrics_by_type': json.dumps(metrics_by_type),
            'metrics_by_stock': json.dumps(metrics_by_stock),
        }
        
        return render(request, 'stocks/model_comparison.html', context)
    
    except Exception as e:
        logger.error(f"Error in model_comparison view: {str(e)}")
        messages.error(request, f"An error occurred while comparing models: {str(e)}")
        return redirect('home')
