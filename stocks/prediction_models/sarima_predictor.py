import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ._base_model import save_model, load_model, MODEL_STORAGE_DIR

logger = logging.getLogger(__name__)

def find_best_sarima_params(y_train, exog=None, max_p=2, max_d=1, max_q=2, 
                           max_P=1, max_D=1, max_Q=1, m=12):
    """
    Grid search for finding optimal SARIMA parameters.
    This replaces pmdarima's auto_arima functionality.
    
    Args:
        y_train: Time series data
        exog: Exogenous variables (optional)
        max_p, max_d, max_q: Maximum values for non-seasonal parameters
        max_P, max_D, max_Q: Maximum values for seasonal parameters
        m: Seasonal period
        
    Returns:
        Best order and seasonal_order based on AIC
    """
    logger.info("Finding optimal SARIMA parameters with grid search...")
    
    # Define parameter ranges
    p = range(0, max_p+1)
    d = range(0, max_d+1)
    q = range(0, max_q+1)
    P = range(0, max_P+1)
    D = range(0, max_D+1)
    Q = range(0, max_Q+1)
    
    # Create a list of all possible parameter combinations
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], m) for x in itertools.product(P, D, Q)]
    
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    
    # Search through parameter combinations
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                # Create and fit SARIMAX model
                model = SARIMAX(
                    y_train,
                    exog=exog,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                model_fit = model.fit(disp=False)
                
                # Update best parameters if current model is better
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = param
                    best_seasonal_order = param_seasonal
                    logger.info(f"New best parameters - Order: {param}, Seasonal Order: {param_seasonal}, AIC: {model_fit.aic}")
            
            except Exception as e:
                continue
    
    logger.info(f"Best SARIMA parameters - Order: {best_order}, Seasonal Order: {best_seasonal_order}, AIC: {best_aic}")
    return best_order, best_seasonal_order

def train_sarima_model(stock_symbol: str, data: pd.DataFrame, target_column: str = None,
                     exog_columns: List[str] = None, test_size: float = 0.2) -> Dict:
    """
    Train a SARIMA model for time series forecasting of stock prices.
    
    Args:
        stock_symbol: Stock symbol for the model
        data: DataFrame with time series data, must be indexed by date
        target_column: Target column to predict (default: stock's Close price)
        exog_columns: List of exogenous variables to include (optional)
        test_size: Proportion of data to use for testing
        
    Returns:
        Dict with trained model, order parameters, and performance metrics
    """
    # Set default target column if not provided
    if target_column is None:
        target_column = f"{stock_symbol}_Close"
    
    logger.info(f"Training SARIMA model for {stock_symbol}...")
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Check if target column exists
    if target_column not in data.columns:
        logger.error(f"Target column '{target_column}' not found in data.")
        return None
    
    # Extract target data and exogenous variables
    y = data[target_column]
    
    # Optional: exogenous variables for SARIMAX
    X = None
    if exog_columns and all(col in data.columns for col in exog_columns):
        X = data[exog_columns]
        logger.info(f"Including {len(exog_columns)} exogenous variables in model.")
    
    # Split into training and test sets
    train_size = int(len(data) * (1 - test_size))
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = (X[:train_size], X[train_size:]) if X is not None else (None, None)
    
    # Create a directory for diagnostics plots
    diagnostics_dir = os.path.join(MODEL_STORAGE_DIR, 'diagnostics', stock_symbol)
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    try:
        # Find optimal SARIMA parameters using our custom function instead of auto_arima
        # Use smaller ranges to keep computation time reasonable
        best_order, best_seasonal_order = find_best_sarima_params(
            y_train, 
            exog=X_train,
            max_p=2, max_d=1, max_q=2,
            max_P=1, max_D=1, max_Q=1,
            m=12  # Typically 12 for monthly seasonality, adjust as needed
        )
        
        # Fit final model with best parameters
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        logger.info("SARIMA model successfully fitted.")
        
        # Make predictions
        y_train_pred = fitted_model.predict(exog=X_train)
        y_test_pred = fitted_model.forecast(steps=len(y_test), exog=X_test)
        
        # Evaluate model
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)  
        test_r2 = r2_score(y_test, y_test_pred)
        
        logger.info(f"Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        logger.info(f"Testing MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        
        # Create and save diagnostic plots
        plt.figure(figsize=(12, 6))
        plt.plot(y_train.index, y_train, label='Training Data')
        plt.plot(y_train.index, y_train_pred, label='Training Predictions')
        plt.plot(y_test.index, y_test, label='Test Data')
        plt.plot(y_test.index, y_test_pred, label='Test Predictions')
        plt.title(f'SARIMA Model for {stock_symbol}')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, f'{stock_symbol}_sarima_fit.png'))
        plt.close()
        
        # Save ACF and PACF plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(y, ax=axes[0])
        plot_pacf(y, ax=axes[1])
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, f'{stock_symbol}_acf_pacf.png'))
        plt.close()
        
        # Create a model dictionary for saving
        model_dict = {
            'model': fitted_model,
            'order': best_order,
            'seasonal_order': best_seasonal_order,
            'exog_columns': exog_columns,
            'target_column': target_column,
            'performance': {
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae), 
                'train_r2': float(train_r2),
                'test_r2': float(test_r2)
            },
            'train_end_date': data.index[train_size - 1].strftime('%Y-%m-%d'),
            'diagnostics_dir': diagnostics_dir,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Special handling for SARIMA model feature importance
        # In SARIMA, we don't have traditional feature importance like in tree-based models,
        # but for exogenous variables, we can examine their p-values
        if exog_columns:
            p_values = fitted_model.pvalues.to_dict()
            # Filter to include only exogenous variables (excluding intercept, AR, MA terms)
            exog_p_values = {k: float(v) for k, v in p_values.items() 
                             if any(col in k for col in exog_columns)}
            
            # Convert p-values to "importance"
            # Lower p-values (more significant) get higher importance
            # Use a simple transformation: 1 - p_value, capped at 0.99
            importance = {k: min(1.0 - v, 0.99) for k, v in exog_p_values.items()}
            model_dict['feature_importance'] = importance
        else:
            # No exogenous variables, so no feature importance
            model_dict['feature_importance'] = {}
        
        # Save the model
        model_path = save_model(model_dict, stock_symbol, 'SARIMA')
        if model_path:
            logger.info(f"SARIMA model for {stock_symbol} saved to {model_path}")
            model_dict['model_path'] = model_path
        else:
            logger.error(f"Failed to save SARIMA model for {stock_symbol}")
        
        return model_dict
    
    except Exception as e:
        logger.error(f"Error training SARIMA model for {stock_symbol}: {str(e)}")
        return None

def predict_next_day(model_path: str, latest_data: pd.DataFrame) -> Optional[float]:
    """
    Make a prediction for the next day's price.
    
    Args:
        model_path: Path to the saved model
        latest_data: DataFrame with latest data for exogenous features
        
    Returns:
        Predicted price for the next day
    """
    try:
        # Load the model
        model_dict = load_model(model_path)
        if not model_dict:
            logger.error(f"Failed to load SARIMA model from {model_path}")
            return None
        
        # Extract components
        model = model_dict['model']
        exog_columns = model_dict['exog_columns']
        
        # Prepare exogenous variables if needed
        exog = None
        if exog_columns and all(col in latest_data.columns for col in exog_columns):
            exog = latest_data[exog_columns].iloc[-1:].values
        
        # Make one-step forecast
        prediction = model.forecast(steps=1, exog=exog)[0]
        
        return float(prediction)
    
    except Exception as e:
        logger.error(f"Error predicting with SARIMA model: {str(e)}")
        return None

def predict_future(model_path: str, latest_data: pd.DataFrame, days: int = 7) -> Optional[List[float]]:
    """
    Make predictions for multiple days into the future.
    
    Args:
        model_path: Path to the saved model
        latest_data: DataFrame with latest data
        days: Number of days to predict
        
    Returns:
        List of predicted prices
    """
    try:
        # Load the model
        model_dict = load_model(model_path)
        if not model_dict:
            logger.error(f"Failed to load SARIMA model from {model_path}")
            return None
        
        # Extract components
        model = model_dict['model']
        exog_columns = model_dict['exog_columns']
        
        # Prepare exogenous variables if needed
        exog = None
        if exog_columns and all(col in latest_data.columns for col in exog_columns):
            # For multi-step forecasting with exogenous variables,
            # we need future values of exogenous variables which we don't have
            # As a simplification, we could use the last known values
            exog = np.repeat(latest_data[exog_columns].iloc[-1:].values, days, axis=0)
        
        # Make multi-step forecast
        predictions = model.forecast(steps=days, exog=exog)
        
        return [float(p) for p in predictions]
    
    except Exception as e:
        logger.error(f"Error making future predictions with SARIMA model: {str(e)}")
        return None
