import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .base_model import BaseModel

class SVMModel(BaseModel):
    """
    Support Vector Machine (SVM) model for stock price prediction.
    Uses Support Vector Regression (SVR) under the hood.
    """
    
    def __init__(self, symbol, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', **kwargs):
        """
        Initialize the SVM model.
        
        Args:
            symbol (str): Stock symbol
            kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C (float): Regularization parameter
            epsilon (float): Epsilon in the epsilon-SVR model
            gamma (str or float): Kernel coefficient
            **kwargs: Additional parameters for the base model
        """
        super().__init__(symbol, **kwargs)
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.metadata['training_parameters'] = {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma
        }
    
    def _create_features(self, data):
        """
        Create features for the SVM model.
        
        Args:
            data (pd.DataFrame): Stock price data
            
        Returns:
            pd.DataFrame: DataFrame with features
        """
        df = data.copy()
        
        # Technical indicators
        
        # Simple moving averages
        df['ma5'] = df['Close'].rolling(window=5).mean()
        df['ma10'] = df['Close'].rolling(window=10).mean()
        df['ma20'] = df['Close'].rolling(window=20).mean()
        
        # Moving average convergence divergence (MACD)
        df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['middle_band'] = df['Close'].rolling(window=20).mean()
        df['std_dev'] = df['Close'].rolling(window=20).std()
        df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
        df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        
        # Price ratios and changes
        df['price_change'] = df['Close'].pct_change()
        df['ma5_ratio'] = df['Close'] / df['ma5']
        df['ma10_ratio'] = df['Close'] / df['ma10']
        df['ma20_ratio'] = df['Close'] / df['ma20']
        
        # Volume features
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma5']
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def preprocess_data(self, data, forecast_horizon=1):
        """
        Preprocess data for SVM model.
        
        Args:
            data (pd.DataFrame): Raw stock data with 'Close' prices
            forecast_horizon (int): Number of days ahead to predict
            
        Returns:
            dict: Processed data including features and targets
        """
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Create features
        feature_df = self._create_features(data)
        
        # Create target variable (future price)
        feature_df['target'] = feature_df['Close'].shift(-forecast_horizon)
        
        # Drop rows with NaN values (due to shifting)
        feature_df = feature_df.dropna()
        
        # Select feature columns (exclude price and volume columns)
        feature_columns = [
            'ma5', 'ma10', 'ma20', 'macd', 'signal', 'rsi', 
            'bb_width', 'price_change', 'ma5_ratio', 'ma10_ratio', 'ma20_ratio',
            'volume_change', 'volume_ratio'
        ]
        
        # Add time-based features if they exist
        if 'day_of_week' in feature_df.columns:
            feature_columns.extend(['day_of_week', 'month', 'quarter'])
        
        # Get features and target
        X = feature_df[feature_columns]
        y = feature_df['target']
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Scale target
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        return {
            'X': X_scaled,
            'y': y_scaled,
            'feature_df': feature_df,
            'feature_columns': feature_columns,
            'last_close': data['Close'].iloc[-1],
            'raw_data': data
        }
    
    def train(self, data, forecast_horizon=1, test_size=0.2, **kwargs):
        """
        Train the SVM model.
        
        Args:
            data (pd.DataFrame): Historical stock data
            forecast_horizon (int): Number of days ahead to predict
            test_size (float): Proportion of data to use for validation
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results and metrics
        """
        # Process the data
        processed = self.preprocess_data(data, forecast_horizon=forecast_horizon)
        X = processed['X']
        y = processed['y']
        
        # Split into training and testing sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create and fit the SVR model
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            **kwargs
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = self.model.predict(X_test)
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Inverse transform to get actual values
        y_test_inv = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
        
        # Mark the model as trained
        self.trained = True
        
        # Update metadata
        self.metadata['updated_at'] = pd.Timestamp.now().isoformat()
        self.metadata['training_parameters'].update({
            'forecast_horizon': forecast_horizon,
            'test_size': test_size,
            **kwargs
        })
        
        self.metadata['data_info'] = {
            'start_date': data.index.min().isoformat(),
            'end_date': data.index.max().isoformat(),
            'num_observations': len(data)
        }
        
        # Store performance metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        self.metadata['performance_metrics'] = metrics
        
        return {
            'model': self.model,
            'metrics': metrics
        }
    
    def predict(self, data, horizon=5, **kwargs):
        """
        Generate stock price predictions.
        
        Args:
            data (pd.DataFrame): Input data for prediction
            horizon (int): Number of days to predict into the future
            **kwargs: Additional prediction parameters
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Process the data
        processed = self.preprocess_data(data)
        feature_columns = processed['feature_columns']
        last_close = processed['last_close']
        
        # Get the most recent feature values
        last_features = processed['feature_df'][feature_columns].iloc[-1:]
        last_features_scaled = self.feature_scaler.transform(last_features)
        
        # Initialize predictions list and the current data for iterative forecasting
        predictions = []
        current_data = data.copy()
        
        for day in range(1, horizon + 1):
            # Get features for the current step
            if day == 1:
                current_features = last_features_scaled
            else:
                # Reprocess the data with the new predictions added
                current_processed = self.preprocess_data(current_data)
                current_features = self.feature_scaler.transform(
                    current_processed['feature_df'][feature_columns].iloc[-1:])
            
            # Make the prediction
            pred_scaled = self.model.predict(current_features)
            
            # Inverse transform to get the actual price
            pred_price = self.target_scaler.inverse_transform(
                pred_scaled.reshape(-1, 1)).flatten()[0]
            
            # Calculate the prediction date (next business day)
            last_date = current_data.index[-1]
            pred_date = last_date + pd.Timedelta(days=1)
            # Skip weekends
            if pred_date.weekday() >= 5:
                pred_date = pred_date + pd.Timedelta(days=2 if pred_date.weekday() == 5 else 1)
            
            # Store the prediction
            predictions.append({
                'date': pred_date,
                'predicted_price': pred_price
            })
            
            # Create a new row with the prediction for the next iteration
            new_row = pd.DataFrame({
                'Open': [pred_price],
                'High': [pred_price * 1.01],  # Estimate
                'Low': [pred_price * 0.99],   # Estimate
                'Close': [pred_price],
                'Volume': [current_data['Volume'].mean()]  # Use average volume
            }, index=[pred_date])
            
            # Append to the current data for the next iteration
            current_data = pd.concat([current_data, new_row])
        
        # Convert the predictions to a DataFrame
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.set_index('date')
        
        # Add percentage change column
        predictions_df['change_pct'] = ((predictions_df['predicted_price'] / last_close) - 1) * 100
        
        return predictions_df
    
    def evaluate(self, test_data, forecast_horizon=1):
        """
        Evaluate the model's performance on test data.
        
        Args:
            test_data (pd.DataFrame): Test data with actual values
            forecast_horizon (int): Forecast horizon used during training
            
        Returns:
            dict: Performance metrics
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before evaluating")
        
        # Process the test data
        processed = self.preprocess_data(test_data, forecast_horizon=forecast_horizon)
        X = processed['X']
        y = processed['y']
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate performance metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Inverse transform to get actual values
        y_inv = self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        y_pred_inv = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_inv - y_pred_inv) / y_inv)) * 100
        
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
