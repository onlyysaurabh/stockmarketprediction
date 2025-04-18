import logging
import os
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, date, timedelta
import pandas as pd
import json

from django.conf import settings
from django.utils import timezone

from .models import Stock, TrainedPredictionModel
from .prediction_data_service import get_prediction_data
from .prediction_models.svm_predictor import train_svm_model, predict_next_day as predict_next_day_svm
from .prediction_models.xgboost_predictor import train_xgboost_model, predict_next_day as predict_next_day_xgboost
from .prediction_models.sarima_predictor import train_sarima_model, predict_next_day as predict_next_day_sarima
from .prediction_models.lstm_predictor import train_lstm_model, predict_next_day as predict_next_day_lstm

logger = logging.getLogger(__name__)

def train_model_for_stock(stock_symbol: str, model_type: str, days: int = 365) -> Optional[Dict[str, Any]]:
    """
    Train a prediction model for a specific stock.
    
    Args:
        stock_symbol: The stock symbol
        model_type: Type of model to train ('SVM', 'XGBOOST', 'SARIMA', 'LSTM')
        days: Number of days of historical data to use
        
    Returns:
        Dictionary with model results and metadata, or None if training failed
    """
    try:
        stock_symbol = stock_symbol.upper()
        logger.info(f"Training {model_type} model for {stock_symbol} using {days} days of data.")
        
        # Check if stock exists
        try:
            stock = Stock.objects.get(symbol=stock_symbol)
        except Stock.DoesNotExist:
            logger.error(f"Stock {stock_symbol} not found in database.")
            return None
        
        # Get prediction data
        data = get_prediction_data(stock_symbol, days=days)
        if data is None or data.empty:
            logger.error(f"Failed to get prediction data for {stock_symbol}.")
            return None
        
        logger.info(f"Got prediction data for {stock_symbol} with shape {data.shape}")
        
        # Train the appropriate model
        model_dict = None
        if model_type == 'SVM':
            model_dict = train_svm_model(stock_symbol, data)
        elif model_type == 'XGBOOST':
            model_dict = train_xgboost_model(stock_symbol, data)
        elif model_type == 'SARIMA':
            model_dict = train_sarima_model(stock_symbol, data)
        elif model_type == 'LSTM':
            model_dict = train_lstm_model(stock_symbol, data)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None
        
        if model_dict is None:
            logger.error(f"Failed to train {model_type} model for {stock_symbol}.")
            return None
        
        # Save model metadata to database
        model_path = model_dict.get('model_path')
        if model_path:
            # Get or create a TrainedPredictionModel entry
            try:
                # Extract performance metrics
                performance = model_dict.get('performance', {})
                
                # Handle feature importance - convert to JSON
                if 'feature_importance' in model_dict:
                    feature_importance = model_dict['feature_importance']
                    # If XGBoost has both native and SHAP importance, use SHAP for consistency
                    if isinstance(feature_importance, dict) and 'shap' in feature_importance:
                        feature_importance = feature_importance['shap']
                else:
                    feature_importance = {}
                    
                # Create model record in database
                trained_model = TrainedPredictionModel.objects.create(
                    stock=stock,
                    model_type=model_type,
                    model_path=model_path,
                    feature_importance=feature_importance
                )
                
                logger.info(f"Saved {model_type} model for {stock_symbol} to database with ID {trained_model.id}")
                
                return {
                    'success': True,
                    'model_id': trained_model.id,
                    'model_path': model_path,
                    'trained_at': trained_model.trained_at,
                    'performance': performance
                }
                
            except Exception as e:
                logger.error(f"Error saving model to database: {e}")
                return {
                    'success': False,
                    'error': f"Error saving model to database: {str(e)}"
                }
        else:
            logger.error(f"No model path found in {model_type} result for {stock_symbol}.")
            return {
                'success': False,
                'error': f"No model path found in {model_type} result for {stock_symbol}."
            }
        
    except Exception as e:
        logger.error(f"Error training {model_type} model for {stock_symbol}: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_prediction_for_stock(stock_symbol: str, model_type: str = None) -> Optional[Dict[str, Any]]:
    """
    Get prediction for a stock using the most recently trained model of specified type.
    If model_type is None, will use the most accurate model.
    
    Args:
        stock_symbol: The stock symbol
        model_type: Type of model to use ('SVM', 'XGBOOST', 'SARIMA', 'LSTM'), or None for best model
        
    Returns:
        Dictionary with prediction and metadata, or None if prediction failed
    """
    try:
        stock_symbol = stock_symbol.upper()
        
        # Check if stock exists
        try:
            stock = Stock.objects.get(symbol=stock_symbol)
        except Stock.DoesNotExist:
            logger.error(f"Stock {stock_symbol} not found in database.")
            return None
        
        # Query the trained models
        if model_type:
            models = TrainedPredictionModel.objects.filter(
                stock=stock, model_type=model_type
            ).order_by('-trained_at')
        else:
            models = TrainedPredictionModel.objects.filter(
                stock=stock
            ).order_by('-trained_at')
            
        if not models.exists():
            logger.error(f"No trained models found for {stock_symbol}" 
                        + (f" of type {model_type}" if model_type else ""))
            return None
        
        # Get the latest model
        latest_model = models.first()
        model_type = latest_model.model_type
        model_path = latest_model.model_path
        
        logger.info(f"Using {model_type} model for {stock_symbol} trained on {latest_model.trained_at}")
        
        # Get latest data for prediction
        data = get_prediction_data(stock_symbol, days=90)  # Get recent data
        if data is None or data.empty:
            logger.error(f"Failed to get recent data for {stock_symbol} prediction.")
            return None
        
        # Make prediction using appropriate model
        prediction = None
        if model_type == 'SVM':
            prediction = predict_next_day_svm(model_path, data)
        elif model_type == 'XGBOOST':
            prediction = predict_next_day_xgboost(model_path, data)
        elif model_type == 'SARIMA':
            prediction = predict_next_day_sarima(model_path, data)
        elif model_type == 'LSTM':
            prediction = predict_next_day_lstm(model_path, data)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None
        
        if prediction is None:
            logger.error(f"Failed to get prediction for {stock_symbol} using {model_type} model.")
            return None
        
        # Calculate prediction date (next trading day)
        last_date = data.index[-1].date()
        prediction_date = last_date + timedelta(days=1)
        # Skip weekends (simple approach)
        while prediction_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            prediction_date += timedelta(days=1)
        
        # Calculate change percentage from the latest price
        current_price = stock.current_price
        if current_price:
            change_pct = ((float(prediction) - float(current_price)) / float(current_price)) * 100
        else:
            change_pct = None
        
        return {
            'stock': stock_symbol,
            'model_type': model_type,
            'prediction': float(prediction),
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'trained_at': latest_model.trained_at.strftime('%Y-%m-%d %H:%M'),
            'model_id': latest_model.id,
            'current_price': float(current_price) if current_price else None,
            'change_pct': change_pct
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction for {stock_symbol}: {e}")
        return None

def get_multiple_predictions(stock_symbol: str, days: int = 7) -> Optional[Dict[str, Any]]:
    """
    Get predictions for a stock for multiple days ahead using all available models.
    
    Args:
        stock_symbol: The stock symbol
        days: Number of days to predict ahead
        
    Returns:
        Dictionary with predictions from each model type, or None if prediction failed
    """
    try:
        stock_symbol = stock_symbol.upper()
        
        # Check if stock exists
        try:
            stock = Stock.objects.get(symbol=stock_symbol)
        except Stock.DoesNotExist:
            logger.error(f"Stock {stock_symbol} not found in database.")
            return None
            
        # Get recent data for prediction
        data = get_prediction_data(stock_symbol, days=90)  # Get recent data
        if data is None or data.empty:
            logger.error(f"Failed to get recent data for {stock_symbol} multi-day prediction.")
            return None
            
        # Get the latest model of each type
        model_predictions = {}
        model_types = ['SVM', 'XGBOOST', 'SARIMA', 'LSTM']
        
        # Generate a date range for the predictions
        last_date = data.index[-1].date()
        dates = []
        
        # Generate business days (crude approximation, in reality would use a calendar of trading days)
        current_date = last_date
        while len(dates) < days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:  # Weekday
                dates.append(current_date.strftime('%Y-%m-%d'))
        
        for model_type in model_types:
            latest_model = TrainedPredictionModel.objects.filter(
                stock=stock, model_type=model_type
            ).order_by('-trained_at').first()
            
            if latest_model:
                logger.info(f"Getting {days}-day prediction for {stock_symbol} with {model_type} model")
                
                # Make multi-day predictions
                # For simplicity, let's use the single-day prediction repeatedly
                # In a real implementation, you would use the predict_future function for each model
                predictions = []
                
                # Make a single prediction and get the prediction function for the model type
                if model_type == 'SVM':
                    from .prediction_models.svm_predictor import predict_future
                elif model_type == 'XGBOOST':
                    from .prediction_models.xgboost_predictor import predict_future
                elif model_type == 'SARIMA':
                    from .prediction_models.sarima_predictor import predict_future
                elif model_type == 'LSTM':
                    from .prediction_models.lstm_predictor import predict_future
                else:
                    continue
                
                # Make the multi-day prediction
                predictions = predict_future(latest_model.model_path, data, days)
                
                if predictions:
                    model_predictions[model_type] = {
                        'model_id': latest_model.id,
                        'trained_at': latest_model.trained_at.strftime('%Y-%m-%d %H:%M'),
                        'predictions': [float(p) for p in predictions],
                    }
        
        if not model_predictions:
            logger.error(f"No predictions were made for {stock_symbol}.")
            return None
            
        # Current price for reference
        current_price = float(stock.current_price) if stock.current_price else None
            
        return {
            'stock': stock_symbol,
            'dates': dates,
            'current_price': current_price,
            'models': model_predictions
        }
        
    except Exception as e:
        logger.error(f"Error getting multi-day predictions for {stock_symbol}: {e}")
        return None