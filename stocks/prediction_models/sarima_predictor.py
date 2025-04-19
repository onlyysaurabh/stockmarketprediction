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
import warnings
import statsmodels.api as sm
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
    
    # Use smart order of parameter combinations - first try simpler models
    # This can save computation time by finding good models earlier
    sorted_pdq = sorted(pdq, key=lambda x: sum(x))
    sorted_seasonal_pdq = sorted(seasonal_pdq, key=lambda x: sum(x[:3]))
    
    # Search through parameter combinations with early stopping if good model found
    early_stop_count = 0
    last_best_aic = float("inf")
    
    # Suppress convergence warnings during grid search
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        for param in sorted_pdq:
            for param_seasonal in sorted_seasonal_pdq:
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
                    
                    model_fit = model.fit(disp=False, maxiter=100)  # Lower max iterations for grid search
                    
                    # Update best parameters if current model is better
                    current_aic = model_fit.aic
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_order = param
                        best_seasonal_order = param_seasonal
                        early_stop_count = 0
                        logger.info(f"New best parameters - Order: {param}, Seasonal Order: {param_seasonal}, AIC: {model_fit.aic}")
                    else:
                        early_stop_count += 1
                    
                    # Early stopping if no improvement for several iterations
                    if early_stop_count > 10:
                        logger.info(f"Early stopping grid search after 10 iterations without improvement")
                        break
                
                except Exception as e:
                    continue
            
            if early_stop_count > 10:
                break
    
    logger.info(f"Best SARIMA parameters - Order: {best_order}, Seasonal Order: {best_seasonal_order}, AIC: {best_aic}")
    return best_order, best_seasonal_order

def check_stationarity(time_series):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.
    
    Args:
        time_series: Time series data
        
    Returns:
        Tuple of (bool, float): (is_stationary, p_value)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(time_series.dropna())
        p_value = result[1]
        is_stationary = p_value <= 0.05  # Stationary if p-value <= 0.05
        logger.info(f"ADF test p-value: {p_value:.6f}, {'Stationary' if is_stationary else 'Non-stationary'}")
        return is_stationary, p_value
    except Exception as e:
        logger.error(f"Error checking stationarity: {str(e)}")
        return False, 1.0

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
    
    # Check stationarity of the target series
    is_stationary, p_value = check_stationarity(y)
    if not is_stationary:
        logger.info("Time series is not stationary. Consider differencing or transformation.")
    
    # Split into training and test sets
    train_size = int(len(data) * (1 - test_size))
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = (X[:train_size], X[train_size:]) if X is not None else (None, None)
    
    # Create a directory for diagnostics plots
    diagnostics_dir = os.path.join(MODEL_STORAGE_DIR, 'diagnostics', stock_symbol)
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    try:
        # Find optimal SARIMA parameters using our custom function
        # For production, you might want to use smaller ranges to keep computation time reasonable
        seasonal_period = 5  # For daily stock data, a week (5 trading days) is often a good season
        
        # Use smaller parameter ranges for faster computation
        # These ranges can be adjusted based on domain knowledge or prior testing
        best_order, best_seasonal_order = find_best_sarima_params(
            y_train, 
            exog=X_train,
            max_p=2, max_d=1, max_q=2,
            max_P=1, max_D=1, max_Q=1,
            m=seasonal_period
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
        
        fitted_model = model.fit(disp=False, maxiter=500)
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
        
        # Save residual plots
        fig = plt.figure(figsize=(12, 8))
        residuals = fitted_model.resid
        
        # Plot residuals
        ax1 = fig.add_subplot(221)
        ax1.plot(residuals)
        ax1.set_title('Residuals')
        
        # Plot residuals histogram
        ax2 = fig.add_subplot(222)
        ax2.hist(residuals, bins=30)
        ax2.set_title('Histogram of Residuals')
        
        # Plot residuals Q-Q plot
        ax3 = fig.add_subplot(223)
        sm.qqplot(residuals, line='45', fit=True, ax=ax3)
        ax3.set_title('Q-Q Plot')
        
        # Plot residuals autocorrelation
        ax4 = fig.add_subplot(224)
        plot_acf(residuals, ax=ax4)
        ax4.set_title('Autocorrelation of Residuals')
        
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, f'{stock_symbol}_sarima_residuals.png'))
        plt.close()
        
        # Calculate feature importance for exogenous variables if available
        feature_importance = {}
        if X is not None and hasattr(fitted_model, 'pvalues') and exog_columns:
            try:
                # Get p-values for exogenous variables (lower p-value = more significant)
                exog_pvalues = fitted_model.pvalues[len(best_order):]
                
                # Convert p-values to importance (1 - p)
                importance_values = 1 - exog_pvalues
                
                # Normalize to sum to 1.0
                total_importance = importance_values.sum()
                if total_importance > 0:
                    normalized_importance = importance_values / total_importance
                    
                    # Map back to feature names
                    for i, col in enumerate(exog_columns):
                        if i < len(normalized_importance):
                            feature_importance[col] = float(normalized_importance[i])
                
                logger.info(f"Feature importance calculated for {len(feature_importance)} exogenous variables.")
            except Exception as e:
                logger.warning(f"Could not calculate feature importance for exogenous variables: {e}")
        
        # Create a model dictionary for saving
        model_dict = {
            'fitted_model': fitted_model,  # The fitted statsmodels SARIMAX object
            'order': best_order,
            'seasonal_order': best_seasonal_order,
            'target_column': target_column,
            'exog_columns': exog_columns,
            'is_stationary': is_stationary,
            'stationarity_p_value': p_value,
            'seasonal_period': seasonal_period,
            'performance': {
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2)
            },
            'feature_importance': feature_importance,
            'summary': str(fitted_model.summary()),
            'diagnostics_dir': diagnostics_dir,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save the model
        model_path = save_model(model_dict, stock_symbol, 'SARIMA')
        if model_path:
            logger.info(f"SARIMA model for {stock_symbol} saved to {model_path}")
            model_dict['model_path'] = model_path
        else:
            logger.error(f"Failed to save SARIMA model for {stock_symbol}")
        
        return model_dict
    
    except Exception as e:
        logger.error(f"Error training SARIMA model: {str(e)}", exc_info=True)
        return None

def predict_next_day(model_path: str, latest_data: pd.DataFrame) -> Optional[float]:
    """
    Make a prediction for the next day's price.
    
    Args:
        model_path: Path to the saved model
        latest_data: DataFrame with latest data for features
        
    Returns:
        Predicted price for the next day
    """
    try:
        # Load the model dictionary
        model_dict = load_model(model_path)
        if not model_dict:
            logger.error(f"Failed to load SARIMA model from {model_path}")
            return None
        
        # Extract model components
        fitted_model = model_dict['fitted_model']
        target_column = model_dict['target_column']
        exog_columns = model_dict.get('exog_columns')
        
        # Prepare exogenous variables if needed
        exog = None
        if exog_columns and len(exog_columns) > 0:
            # Verify all required columns exist
            missing_cols = [col for col in exog_columns if col not in latest_data.columns]
            if missing_cols:
                logger.error(f"Missing exogenous columns for SARIMA prediction: {missing_cols}")
                return None
            
            # Get latest values for exogenous variables
            exog = latest_data[exog_columns].iloc[-1:].values  # Need a 2D array for exog
        
        # Forecast one step ahead
        prediction = fitted_model.forecast(steps=1, exog=exog)[0]
        
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
        days: Number of days to forecast
        
    Returns:
        List of predicted prices
    """
    try:
        # Load the model dictionary
        model_dict = load_model(model_path)
        if not model_dict:
            logger.error(f"Failed to load SARIMA model from {model_path}")
            return None
        
        # Extract model components
        fitted_model = model_dict['fitted_model']
        target_column = model_dict['target_column']
        exog_columns = model_dict.get('exog_columns')
        
        # Prepare exogenous variables for multiple steps if needed
        exog = None
        
        if exog_columns and len(exog_columns) > 0:
            # This gets tricky because we need future exogenous values
            # For simplicity, one approach is to use the last values
            logger.warning("Using last available values for all future exogenous variables. Consider using more sophisticated forecasting for exogenous variables.")
            # Repeat the last row of exogenous variables for each future step
            exog = np.repeat(latest_data[exog_columns].iloc[-1:].values, days, axis=0)
        
        # Forecast multiple steps ahead
        predictions = fitted_model.forecast(steps=days, exog=exog)
        
        # Convert to list of floats
        predictions = [float(x) for x in predictions]
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error making future predictions with SARIMA model: {str(e)}")
        return None
