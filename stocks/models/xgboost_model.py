import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost model for stock price prediction using gradient boosted trees.
    """
    
    def __init__(self, symbol, n_estimators=100, learning_rate=0.1, max_depth=5, **kwargs):
        """
        Initialize the XGBoost model.
        
        Args:
            symbol (str): Stock symbol
            n_estimators (int): Number of boosting rounds
            learning_rate (float): Learning rate/eta
            max_depth (int): Maximum tree depth
            **kwargs: Additional parameters for the base model
        """
        super().__init__(symbol, **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.scaler = StandardScaler()
        
        self.metadata['training_parameters'] = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
        }
    
    def _create_features(self, data):
        """
        Create features for the XGBoost model.
        
        Args:
            data (pd.DataFrame): Stock price data
            
        Returns:
            pd.DataFrame: DataFrame with features
        """
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volume features
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma5'] = df['Volume'].rolling(window=5).mean()
        
        # Moving averages
        df['ma5'] = df['Close'].rolling(window=5).mean()
        df['ma10'] = df['Close'].rolling(window=10).mean()
        df['ma20'] = df['Close'].rolling(window=20).mean()
        df['ma50'] = df['Close'].rolling(window=50).mean()
        
        # Moving average ratios
        df['ma5_ratio'] = df['Close'] / df['ma5']
        df['ma10_ratio'] = df['Close'] / df['ma10']
        df['ma20_ratio'] = df['Close'] / df['ma20']
        
        # Volatility
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        
        # Price differences
        df['diff_1'] = df['Close'].diff(1)
        df['diff_5'] = df['Close'].diff(5)
        
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # MACD (Moving Average Convergence Divergence)
        macd, signal, hist = self._calculate_macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Date-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, series, window=14):
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            series (pd.Series): Price series
            window (int): RSI calculation window
            
        Returns:
            pd.Series: RSI values
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, series, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            series (pd.Series): Price series
            fast (int): Fast period
            slow (int): Slow period
            signal (int): Signal period
            
        Returns:
            tuple: MACD line, signal line, and histogram
        """
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    def _create_target(self, data, horizon=1):
        """
        Create the target variable for the model.
        
        Args:
            data (pd.DataFrame): Stock price data
            horizon (int): Prediction horizon in days
            
        Returns:
            pd.Series: Target variable
        """
        # The target is the future price change over the horizon
        target = data['Close'].shift(-horizon) / data['Close'] - 1
        
        return target
    
    def preprocess_data(self, data, horizon=1):
        """
        Preprocess data for XGBoost model.
        
        Args:
            data (pd.DataFrame): Raw stock data with price and volume
            horizon (int): Prediction horizon in days
            
        Returns:
            dict: Processed data including features and targets
        """
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Create features
        feature_df = self._create_features(data)
        
        # Create target variable
        target = self._create_target(feature_df, horizon=horizon)
        
        # Drop Close price and other non-feature columns
        feature_cols = [col for col in feature_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close']]
        features = feature_df[feature_cols]
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(features)
        scaled_features_df = pd.DataFrame(
            scaled_features, 
            index=features.index, 
            columns=features.columns
        )
        
        # Create final dataframe
        processed_df = pd.concat([scaled_features_df, target], axis=1)
        processed_df.columns = list(scaled_features_df.columns) + ['target']
        
        # Drop rows with NaN values
        processed_df = processed_df.dropna()
        
        return {
            'processed_df': processed_df,
            'features': scaled_features_df,
            'target': target,
            'raw_data': data,
            'last_close_price': data['Close'].iloc[-1]
        }
    
    def train(self, data, horizon=1, early_stopping_rounds=10, test_size=0.2, **kwargs):
        """
        Train the XGBoost model.
        
        Args:
            data (pd.DataFrame): Historical stock data
            horizon (int): Prediction horizon in days
            early_stopping_rounds (int): Early stopping parameter for XGBoost
            test_size (float): Proportion of data to use for validation
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results and metrics
        """
        # Process the data
        processed = self.preprocess_data(data, horizon=horizon)
        df = processed['processed_df']
        
        # Split features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split into train and validation sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        
        # Create the model
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            **kwargs
        )
        
        # Train the model
        eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='rmse',
            early_stopping_rounds=early_stopping_rounds,
            verbose=True
        )
        
        # Make predictions on validation set
        y_pred = self.model.predict(X_val)
        
        # Calculate performance metrics
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        
        # Mark the model as trained
        self.trained = True
        
        # Update metadata
        self.metadata['updated_at'] = pd.Timestamp.now().isoformat()
        self.metadata['training_parameters'].update({
            'horizon': horizon,
            'test_size': test_size,
            'best_iteration': self.model.best_iteration,
            **kwargs
        })
        
        self.metadata['data_info'] = {
            'start_date': data.index.min().isoformat(),
            'end_date': data.index.max().isoformat(),
            'num_observations': len(data)
        }
        
        # Get feature importances
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': self.model,
            'rmse': rmse,
            'feature_importance': importance_df,
            'best_iteration': self.model.best_iteration
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
        
        # Process the latest data
        processed = self.preprocess_data(data, horizon=1)  # Use horizon=1 for initial processing
        last_close_price = processed['last_close_price']
        
        # Make recursive predictions for the horizon
        predictions = []
        current_data = data.copy()
        
        for step in range(1, horizon + 1):
            # Preprocess the current data
            step_processed = self.preprocess_data(current_data, horizon=1)
            step_features = step_processed['features'].iloc[-1:]  # Get the latest feature row
            
            # Predict the price change
            price_change_pct = self.model.predict(step_features)[0]
            
            # Calculate the predicted price
            if step == 1:
                predicted_price = last_close_price * (1 + price_change_pct)
            else:
                predicted_price = current_data['Close'].iloc[-1] * (1 + price_change_pct)
            
            # Generate the prediction date (assuming business days)
            last_date = current_data.index[-1]
            pred_date = last_date + pd.Timedelta(days=1)
            if pred_date.weekday() >= 5:  # Skip weekends
                pred_date = pred_date + pd.Timedelta(days=2 if pred_date.weekday() == 5 else 1)
            
            predictions.append({
                'date': pred_date,
                'predicted_price': predicted_price,
                'predicted_change_pct': price_change_pct
            })
            
            # Update the current data with the prediction
            new_row = current_data.iloc[-1:].copy()
            new_row.index = [pred_date]
            new_row['Close'] = predicted_price
            
            # Simple estimation for other price columns
            change_ratio = predicted_price / current_data['Close'].iloc[-1]
            new_row['Open'] = current_data['Open'].iloc[-1] * change_ratio
            new_row['High'] = current_data['High'].iloc[-1] * change_ratio
            new_row['Low'] = current_data['Low'].iloc[-1] * change_ratio
            
            # Append the new row to the current data
            current_data = pd.concat([current_data, new_row])
        
        # Create a DataFrame with the predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.set_index('date')
        
        return predictions_df[['predicted_price', 'predicted_change_pct']]
    
    def evaluate(self, test_data, horizon=1):
        """
        Evaluate the model's performance on test data.
        
        Args:
            test_data (pd.DataFrame): Test data with actual values
            horizon (int): Prediction horizon in days
            
        Returns:
            dict: Performance metrics
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before evaluating")
        
        # Process the test data
        processed = self.preprocess_data(test_data, horizon=horizon)
        df = processed['processed_df']
        
        # Split features and target
        X = df.drop('target', axis=1)
        y_true = df['target']
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate performance metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
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
