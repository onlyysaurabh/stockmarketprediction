import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .base_model import BaseModel

class SARIMAXModel(BaseModel):
    """
    SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) model
    for time series stock price prediction.
    """
    
    def __init__(self, symbol, order=None, seasonal_order=None, **kwargs):
        """
        Initialize the SARIMAX model.
        
        Args:
            symbol (str): Stock symbol
            order (tuple, optional): ARIMA order (p, d, q)
            seasonal_order (tuple, optional): Seasonal order (P, D, Q, s)
            **kwargs: Additional parameters for the base model
        """
        super().__init__(symbol, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.metadata['training_parameters'] = {
            'order': order,
            'seasonal_order': seasonal_order,
        }
    
    def preprocess_data(self, data):
        """
        Preprocess data for SARIMAX model.
        
        Args:
            data (pd.DataFrame): Raw stock data with 'Close' prices
            
        Returns:
            pd.DataFrame: Processed data
        """
        # Ensure data is sorted by date
        processed_data = data.sort_index()
        
        # Add date-based features that might help with seasonality
        if isinstance(processed_data.index, pd.DatetimeIndex):
            processed_data['day_of_week'] = processed_data.index.dayofweek
            processed_data['month'] = processed_data.index.month
            processed_data['quarter'] = processed_data.index.quarter
        
        # Check for missing values
        if processed_data['Close'].isnull().sum() > 0:
            # Fill missing values using forward fill
            processed_data = processed_data.fillna(method='ffill')
            
        return processed_data
    
    def find_best_parameters(self, data, max_p=5, max_d=2, max_q=5, seasonal=True):
        """
        Find the best SARIMAX parameters using auto_arima.
        
        Args:
            data (pd.DataFrame): Historical stock data
            max_p (int): Maximum p value to consider
            max_d (int): Maximum d value to consider
            max_q (int): Maximum q value to consider
            seasonal (bool): Whether to include seasonal components
            
        Returns:
            tuple: Best order and seasonal_order parameters
        """
        # Extract the 'Close' price series
        series = data['Close']
        
        # Use auto_arima to find the best parameters
        model = auto_arima(
            series,
            start_p=1, start_q=1, start_P=1, start_Q=1,
            max_p=max_p, max_d=max_d, max_q=max_q,
            m=5 if seasonal else 1,  # m=5 for weekly seasonality in stock data
            seasonal=seasonal,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        order = model.order
        seasonal_order = model.seasonal_order if seasonal else None
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.metadata['training_parameters']['order'] = order
        self.metadata['training_parameters']['seasonal_order'] = seasonal_order
        
        return order, seasonal_order
    
    def train(self, data, exog_variables=None, auto_tune=False, **kwargs):
        """
        Train the SARIMAX model.
        
        Args:
            data (pd.DataFrame): Historical stock data
            exog_variables (list): List of exogenous variable names to include
            auto_tune (bool): Whether to automatically find the best parameters
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results and metrics
        """
        processed_data = self.preprocess_data(data)
        
        # Extract target variable (Close price)
        endog = processed_data['Close']
        
        # Extract exogenous variables if specified
        exog = None
        if exog_variables:
            exog = processed_data[exog_variables]
        
        # Find optimal parameters if auto_tune is True
        if auto_tune:
            self.find_best_parameters(processed_data)
        
        # Use default parameters if none are specified
        if self.order is None:
            self.order = (1, 1, 1)
            self.metadata['training_parameters']['order'] = self.order
            
        if self.seasonal_order is None:
            self.seasonal_order = (0, 0, 0, 0)
            self.metadata['training_parameters']['seasonal_order'] = self.seasonal_order
        
        # Create and fit the SARIMAX model
        self.model = SARIMAX(
            endog=endog,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            **kwargs
        )
        
        self.fit_result = self.model.fit(disp=False)
        self.trained = True
        
        # Update metadata
        self.metadata['updated_at'] = pd.Timestamp.now().isoformat()
        self.metadata['training_parameters'].update(kwargs)
        self.metadata['data_info'] = {
            'start_date': processed_data.index.min().isoformat(),
            'end_date': processed_data.index.max().isoformat(),
            'num_observations': len(processed_data)
        }
        
        return {
            'model': self.model,
            'fit_result': self.fit_result,
            'aic': self.fit_result.aic,
            'bic': self.fit_result.bic
        }
    
    def predict(self, data, horizon=5, exog_variables=None, **kwargs):
        """
        Generate stock price predictions.
        
        Args:
            data (pd.DataFrame): Input data for prediction
            horizon (int): Number of days to predict
            exog_variables (list): List of exogenous variables
            **kwargs: Additional prediction parameters
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if not self.trained or self.fit_result is None:
            raise ValueError("Model must be trained before making predictions")
        
        processed_data = self.preprocess_data(data)
        
        # Prepare exogenous variables
        exog_future = None
        if exog_variables:
            if len(processed_data) < horizon:
                raise ValueError("Not enough exogenous data for the prediction horizon")
            exog_future = processed_data[exog_variables][-horizon:]
        
        # Get the last observation date
        last_date = processed_data.index[-1]
        
        # Generate future dates for the prediction horizon
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='B'  # Business day frequency
        )
        
        # Make the forecast
        forecast = self.fit_result.get_forecast(steps=horizon, exog=exog_future)
        
        # Get the predicted values and confidence intervals
        pred_mean = forecast.predicted_mean
        pred_ci = forecast.conf_int()
        
        # Create a DataFrame with the predictions
        predictions = pd.DataFrame({
            'Predicted_Close': pred_mean,
            'Lower_CI': pred_ci.iloc[:, 0],
            'Upper_CI': pred_ci.iloc[:, 1]
        }, index=future_dates)
        
        return predictions
    
    def evaluate(self, test_data, exog_variables=None):
        """
        Evaluate the model's performance on test data.
        
        Args:
            test_data (pd.DataFrame): Test data with actual values
            exog_variables (list): List of exogenous variables
            
        Returns:
            dict: Performance metrics
        """
        if not self.trained or self.fit_result is None:
            raise ValueError("Model must be trained before evaluating")
        
        processed_data = self.preprocess_data(test_data)
        
        # Extract actual values
        actual = processed_data['Close']
        
        # Prepare exogenous variables
        exog = None
        if exog_variables:
            exog = processed_data[exog_variables]
        
        # Get one-step-ahead predictions for the test period
        predictions = self.fit_result.get_prediction(
            start=processed_data.index[0],
            end=processed_data.index[-1],
            exog=exog
        )
        
        pred_mean = predictions.predicted_mean
        
        # Calculate performance metrics
        mse = mean_squared_error(actual, pred_mean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, pred_mean)
        r2 = r2_score(actual, pred_mean)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - pred_mean) / actual)) * 100
        
        # Store metrics in metadata
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        self.metadata['performance_metrics'] = metrics
        
        return metrics
