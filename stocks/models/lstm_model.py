import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from .base_model import BaseModel

class LSTMModel(BaseModel):
    """
    Long Short-Term Memory (LSTM) neural network model for stock price prediction.
    """
    
    def __init__(self, symbol, sequence_length=10, units=50, layers=1, dropout=0.2, **kwargs):
        """
        Initialize the LSTM model.
        
        Args:
            symbol (str): Stock symbol
            sequence_length (int): Number of time steps to look back
            units (int): Number of LSTM units per layer
            layers (int): Number of LSTM layers
            dropout (float): Dropout rate for regularization
            **kwargs: Additional parameters for the base model
        """
        super().__init__(symbol, **kwargs)
        self.sequence_length = sequence_length
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.metadata['training_parameters'] = {
            'sequence_length': sequence_length,
            'units': units,
            'layers': layers,
            'dropout': dropout,
        }
    
    def preprocess_data(self, data):
        """
        Preprocess data for LSTM model.
        
        Args:
            data (pd.DataFrame): Raw stock data with 'Close' prices
            
        Returns:
            dict: Processed data including scaled values and original DataFrame
        """
        # Ensure data is sorted by date
        processed_data = data.sort_index()
        
        # Check for missing values
        if processed_data['Close'].isnull().sum() > 0:
            # Fill missing values using forward fill
            processed_data = processed_data.fillna(method='ffill')
        
        # Scale the 'Close' prices to [0, 1] range
        close_values = processed_data['Close'].values.reshape(-1, 1)
        scaled_close = self.scaler.fit_transform(close_values)
        
        # Add additional features (technical indicators) if needed
        processed_data['Returns'] = processed_data['Close'].pct_change()
        processed_data['MA5'] = processed_data['Close'].rolling(window=5).mean()
        processed_data['MA20'] = processed_data['Close'].rolling(window=20).mean()
        processed_data['RSI'] = self._calculate_rsi(processed_data['Close'])
        
        # Drop NaN values resulting from the calculations
        processed_data = processed_data.dropna()
        
        return {
            'processed_df': processed_data,
            'scaled_close': scaled_close,
            'raw_data': data
        }
    
    def _calculate_rsi(self, series, period=14):
        """
        Calculate the Relative Strength Index (RSI) technical indicator.
        
        Args:
            series (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _create_sequences(self, scaled_data, sequence_length=None):
        """
        Create sequences of data for LSTM input.
        
        Args:
            scaled_data (np.array): Scaled feature data
            sequence_length (int, optional): Length of each sequence
            
        Returns:
            tuple: X sequences and y targets
        """
        seq_length = sequence_length or self.sequence_length
        X, y = [], []
        
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i + seq_length])
            y.append(scaled_data[i + seq_length, 0])
            
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape):
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.units,
            return_sequences=self.layers > 1,
            input_shape=input_shape,
            recurrent_dropout=self.dropout
        ))
        model.add(Dropout(self.dropout))
        
        # Add intermediate LSTM layers if specified
        for _ in range(self.layers - 2):
            model.add(LSTM(
                units=self.units,
                return_sequences=True,
                recurrent_dropout=self.dropout
            ))
            model.add(Dropout(self.dropout))
        
        # Final LSTM layer if we have more than one layer
        if self.layers > 1:
            model.add(LSTM(units=self.units, recurrent_dropout=self.dropout))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train(self, data, validation_split=0.2, epochs=100, batch_size=32, patience=10, **kwargs):
        """
        Train the LSTM model.
        
        Args:
            data (pd.DataFrame): Historical stock data
            validation_split (float): Portion of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            patience (int): Patience for early stopping
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results and metrics
        """
        # Process the data
        processed = self.preprocess_data(data)
        scaled_data = processed['scaled_close']
        
        # Create sequences for training
        X, y = self._create_sequences(scaled_data)
        
        # Reshape X for LSTM input [samples, time steps, features]
        input_shape = (X.shape[1], X.shape[2])
        
        # Split into training and validation sets
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build the model
        self.model = self._build_model(input_shape)
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Mark the model as trained
        self.trained = True
        
        # Update metadata
        self.metadata['updated_at'] = pd.Timestamp.now().isoformat()
        self.metadata['training_parameters'].update({
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'final_epoch': len(history.history['loss']),
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        self.metadata['data_info'] = {
            'start_date': data.index.min().isoformat(),
            'end_date': data.index.max().isoformat(),
            'num_observations': len(data)
        }
        
        return {
            'model': self.model,
            'history': history.history,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
    
    def predict(self, data, horizon=5, **kwargs):
        """
        Generate stock price predictions.
        
        Args:
            data (pd.DataFrame): Input data for prediction
            horizon (int): Number of days to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Process the data
        processed = self.preprocess_data(data)
        scaled_data = processed['scaled_close']
        
        # Get the last sequence from the data
        last_sequence = scaled_data[-self.sequence_length:]
        
        # Make predictions for the horizon
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(horizon):
            # Reshape for prediction [samples, time steps, features]
            X = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict the next value
            next_pred = self.model.predict(X, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update the sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
            current_sequence = current_sequence.reshape(-1, 1)
        
        # Inverse transform the predictions to get the actual prices
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Generate future dates for the prediction horizon
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='B'  # Business day frequency
        )
        
        # Create a DataFrame with the predictions
        predictions_df = pd.DataFrame({
            'Predicted_Close': predictions.flatten()
        }, index=future_dates)
        
        return predictions_df
    
    def evaluate(self, test_data):
        """
        Evaluate the model's performance on test data.
        
        Args:
            test_data (pd.DataFrame): Test data with actual values
            
        Returns:
            dict: Performance metrics
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before evaluating")
        
        # Process the test data
        processed = self.preprocess_data(test_data)
        scaled_data = processed['scaled_close']
        
        # Create sequences for testing
        X_test, y_test = self._create_sequences(scaled_data)
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Inverse transform the predictions and actual values
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
        
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
