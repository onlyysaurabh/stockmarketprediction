import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ._base_model import save_model, load_model, MODEL_STORAGE_DIR

# Assuming _base_model.py is in the same directory or accessible via Python path
try:
    # LSTM might use save/load from base, but needs specific handling for Keras models
    # from ._base_model import calculate_feature_importance # Use base for importance calc (SHAP needed) - Not used currently
    # Keras models have their own save/load methods, usually preferred over joblib
    pass # Keep try/except structure
except ImportError:
    logging.warning("Could not import from _base_model.")
    def calculate_feature_importance(*args, **kwargs): return None


logger = logging.getLogger(__name__)

# Define features and target
TARGET_COLUMN = 'Target_Close' # Predict next day's closing price
# Example features (adjust based on data exploration and feature engineering)
# LSTM can potentially handle more raw features, but scaling is crucial
FEATURES = [
    'AAPL_Close', 'AAPL_Open', 'AAPL_High', 'AAPL_Low', 'AAPL_Volume', # Stock features
    'GC=F_Close', 'CL=F_Close', '^GSPC_Close', # Commodity/Index prices
    'avg_positive', 'avg_negative', 'avg_neutral', 'news_count' # Sentiment features
]
# Note: Feature list needs refinement based on actual data from prediction_data_service.

def create_lstm_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of data for LSTM training.
    Assumes the last column in 'data' is the target variable.

    Args:
        data (np.ndarray): Scaled data array (features + target).
        sequence_length (int): The number of time steps in each sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (sequences of features) and y (target values).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Sequence: data from i to i + sequence_length
        seq = data[i:(i + sequence_length)]
        # Target: the target value at the end of the sequence
        target = data[i + sequence_length, -1] # Assumes target is the last column
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def train_lstm(data: pd.DataFrame, stock_symbol: str, sequence_length: int = 60) -> Tuple[Optional[Any], Optional[MinMaxScaler], Optional[Dict[str, float]]]:
    """
    Trains an LSTM model on the prepared data.

    Args:
        data (pd.DataFrame): DataFrame from prediction_data_service.
        stock_symbol (str): The stock symbol.
        sequence_length (int): Number of past time steps to use for prediction.

    Returns:
        Tuple[Optional[keras.Model], Optional[MinMaxScaler], Optional[Dict[str, float]]]:
            - The trained Keras LSTM model.
            - The scaler used for the target variable (needed for inverse transform).
            - Feature importance dictionary (placeholder for LSTM, SHAP needed).
            Returns (None, None, None) if training fails.
    """
    logger.info(f"Starting LSTM training for {stock_symbol}...")
    close_col = f'{stock_symbol}_Close'

    # 1. Prepare Data & Feature Selection
    # Create target column (next day's close)
    data = data.copy()
    data[TARGET_COLUMN] = data[close_col].shift(-1)
    data.dropna(inplace=True) # Drop last row with NaN target

    # Select features and target
    lstm_features = [f'{stock_symbol}_Close', f'{stock_symbol}_Open', f'{stock_symbol}_High', f'{stock_symbol}_Low', f'{stock_symbol}_Volume'] + \
                    [col for col in data.columns if col.endswith('_Close') and not col.startswith(stock_symbol)] + \
                    ['avg_positive', 'avg_negative', 'avg_neutral', 'news_count']
    
    final_features = [f for f in lstm_features if f in data.columns]
    if not final_features:
        logger.error("No valid features found for LSTM.")
        return None, None, None
        
    if TARGET_COLUMN not in data.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found for LSTM.")
        return None, None, None
        
    logger.info(f"Using features for LSTM: {final_features}")
    
    # Include target in data for scaling and sequencing
    data_to_scale = data[final_features + [TARGET_COLUMN]].copy()
    
    if data_to_scale.isnull().any().any():
        logger.warning(f"Data for LSTM contains NaNs before scaling. Filling with 0 for {stock_symbol}.")
        data_to_scale = data_to_scale.fillna(0) # Simple imputation

    if data_to_scale.empty:
        logger.error("Data is empty before scaling for LSTM.")
        return None, None, None

    # 2. Scale Data
    # Scale all features and target together initially
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_scale)

    # Separate scaler for the target variable (needed for inverse transform later)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(data[[TARGET_COLUMN]]) # Fit on the original target column

    # 3. Create Sequences
    X, y = create_lstm_sequences(scaled_data, sequence_length)

    if X.shape[0] == 0:
        logger.error(f"Not enough data to create sequences of length {sequence_length} for {stock_symbol}.")
        return None, None, None

    # Reshape X for LSTM input [samples, time steps, features]
    # X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2])) # Already in correct shape from create_lstm_sequences if data includes target

    # 4. Build LSTM Model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1) # Output layer: 1 node for predicting the target value
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.info("LSTM model compiled.")
    model.summary(print_fn=logger.info)

    # 5. Train Model
    # Use early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    try:
        history = model.fit(
            X, y,
            epochs=50, # Adjust epochs
            batch_size=32, # Adjust batch size
            callbacks=[early_stopping],
            verbose=0 # Set to 1 or 2 for more verbose output during training
        )
        logger.info(f"LSTM model trained successfully for {stock_symbol}. Final loss: {history.history['loss'][-1]}")
    except Exception as e:
        logger.error(f"Error during LSTM model fitting for {stock_symbol}: {e}", exc_info=True)
        return None, None, None

    # 6. Feature Importance (Placeholder - requires SHAP for LSTMs)
    # SHAP for LSTMs (e.g., DeepExplainer) can be complex to set up correctly
    # importance = calculate_feature_importance(model, 'LSTM', X) # Pass appropriate data for SHAP
    importance = {"placeholder_lstm": 1.0} # Dummy importance

    return model, target_scaler, importance


def predict_lstm(model: Any, data: pd.DataFrame, stock_symbol: str, sequence_length: int, target_scaler: MinMaxScaler) -> Optional[pd.DataFrame]:
    """
    Makes future predictions using a trained LSTM model.

    Args:
        model: The trained Keras LSTM model.
        data (pd.DataFrame): DataFrame with recent historical data (at least sequence_length).
        stock_symbol (str): The stock symbol.
        sequence_length (int): Number of past time steps used for prediction.
        target_scaler (MinMaxScaler): Scaler fitted on the target variable.

    Returns:
        Optional[pd.DataFrame]: DataFrame with the next step prediction. Includes 'Date' and 'Prediction'.
                                Returns None if prediction fails.
    """
    logger.info(f"Making predictions with LSTM for {stock_symbol}...")

    # 1. Prepare Input Data
    # close_col = f'{stock_symbol}_Close' # Unused variable
    lstm_features = [f'{stock_symbol}_Close', f'{stock_symbol}_Open', f'{stock_symbol}_High', f'{stock_symbol}_Low', f'{stock_symbol}_Volume'] + \
                    [col for col in data.columns if col.endswith('_Close') and not col.startswith(stock_symbol)] + \
                    ['avg_positive', 'avg_negative', 'avg_neutral', 'news_count']
    
    final_features = [f for f in lstm_features if f in data.columns]
    if not final_features:
        logger.error("No valid features found for LSTM prediction.")
        return None

    # Select the last 'sequence_length' data points for prediction input
    # Need features + a placeholder for the target column for scaling consistency
    input_data = data[final_features].iloc[-sequence_length:].copy()
    
    # Add a temporary target column (its value doesn't matter for scaling input features)
    input_data['Target_Placeholder'] = 0 
    
    if input_data.isnull().any().any():
        logger.warning(f"Prediction input data for LSTM contains NaNs. Filling with 0 for {stock_symbol}.")
        input_data = input_data.fillna(0)

    if len(input_data) < sequence_length:
        logger.error(f"Not enough data ({len(input_data)}) to form prediction sequence of length {sequence_length}.")
        return None

    # 2. Scale Input Data
    # Use the same scaler used during training (fit on features + target)
    # Recreate the scaler based on training data distribution (or save/load the scaler)
    # For simplicity, assume we refit the scaler here - THIS IS NOT IDEAL FOR PRODUCTION
    # Ideally, save the scaler used in training and load it here.
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler on the historical data used for training (or a representative sample)
    # This requires access to the original training data or saving the scaler state.
    # Simplified approach (less accurate): fit on the available recent data + placeholder target
    scaled_input_data = scaler.fit_transform(input_data)

    # Remove the placeholder target column after scaling
    scaled_input_sequence = scaled_input_data[:, :-1] 

    # Reshape for LSTM [1, time steps, features]
    X_pred = np.reshape(scaled_input_sequence, (1, sequence_length, len(final_features)))

    # 3. Make Prediction
    try:
        predicted_scaled = model.predict(X_pred, verbose=0)
        
        # 4. Inverse Transform the Prediction
        # Use the scaler that was specifically fit on the TARGET variable during training
        predicted_value = target_scaler.inverse_transform(predicted_scaled)

        # Create DataFrame for the result
        # Predict for the next business day after the last date in the input data
        last_date = data.index[-1]
        prediction_date = last_date + pd.Timedelta(days=1)
        # Adjust for weekends/holidays if necessary, e.g., using pandas BDay
        # prediction_date = last_date + pd.offsets.BDay(1) 

        results_df = pd.DataFrame({'Prediction': predicted_value[0]}, index=[prediction_date])
        results_df.index.name = 'Date'

        logger.info(f"LSTM prediction generated successfully for {stock_symbol} for date {prediction_date.date()}.")
        return results_df

    except Exception as e:
        logger.error(f"Error during LSTM prediction for {stock_symbol}: {e}", exc_info=True)
        return None

# Note: Saving/Loading Keras models
# model.save('lstm_model.h5') # Saves architecture, weights, optimizer state
# from tensorflow.keras.models import load_model
# loaded_model = load_model('lstm_model.h5')
# Need to handle saving/loading scalers separately (e.g., using joblib)

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and targets for LSTM training.
    
    Args:
        data: Input data array
        seq_length: Length of input sequences
        
    Returns:
        Tuple of (X, y) where X is the input sequences and y is the target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Target is the first column (price)
    return np.array(X), np.array(y)

def train_lstm_model(stock_symbol: str, data: pd.DataFrame, target_column: str = None,
                    feature_columns: List[str] = None, test_size: float = 0.2,
                    sequence_length: int = 30) -> Dict:
    """
    Train an LSTM model for stock price prediction.
    
    Args:
        stock_symbol: Stock symbol for the model
        data: DataFrame with features and target, must be indexed by date
        target_column: Column to predict (default: stock's Close price)
        feature_columns: List of feature column names to include
        test_size: Proportion of data to use for testing
        sequence_length: Number of time steps to use for each prediction
        
    Returns:
        Dict with trained model, scalers, and performance metrics
    """
    try:
        # Set default target column if not provided
        if target_column is None:
            target_column = f"{stock_symbol}_Close"
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Set default feature columns if not provided
        if feature_columns is None:
            # For LSTM, we often want to include price-related columns of the target stock
            feature_columns = [col for col in data.columns 
                               if any(price_type in col for price_type in 
                                      ['Open', 'High', 'Low', 'Close', 'Volume'])]
        
        logger.info(f"Training LSTM model for {stock_symbol} with {len(feature_columns)} features.")
        
        # Check if target column exists in features, if not, add it
        if target_column not in feature_columns:
            feature_columns.append(target_column)
        
        # Extract features and ensure target column is the first column
        feature_columns.remove(target_column)
        feature_columns = [target_column] + feature_columns  # Target is first
        
        # Extract selected features
        dataset = data[feature_columns].values
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Create a separate scaler for the target column for easy inverse transform
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_data = data[[target_column]].values
        target_scaler.fit(target_data)
        
        # Create sequences
        X, y = create_sequences(scaled_data, sequence_length)
        
        # Split into training and testing sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build the LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        # Create a directory for model checkpoints and diagnostics
        model_dir = os.path.join(MODEL_STORAGE_DIR, 'keras_models', stock_symbol)
        os.makedirs(model_dir, exist_ok=True)
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, f"{stock_symbol}_lstm_checkpoint.h5"),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history plot
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'LSTM Training History for {stock_symbol}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.savefig(os.path.join(model_dir, f'{stock_symbol}_lstm_training.png'))
        plt.close()
        
        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Inverse transform to get actual prices
        train_predict_actual = np.zeros((train_predict.shape[0], dataset.shape[1]))
        train_predict_actual[:, 0] = train_predict.flatten()
        train_predict_actual = scaler.inverse_transform(train_predict_actual)[:, 0]
        
        test_predict_actual = np.zeros((test_predict.shape[0], dataset.shape[1]))
        test_predict_actual[:, 0] = test_predict.flatten()
        test_predict_actual = scaler.inverse_transform(test_predict_actual)[:, 0]
        
        # Get actual target values
        y_train_actual = np.zeros((y_train.shape[0], dataset.shape[1]))
        y_train_actual[:, 0] = y_train
        y_train_actual = scaler.inverse_transform(y_train_actual)[:, 0]
        
        y_test_actual = np.zeros((y_test.shape[0], dataset.shape[1]))
        y_test_actual[:, 0] = y_test
        y_test_actual = scaler.inverse_transform(y_test_actual)[:, 0]
        
        # Calculate performance metrics
        train_mse = mean_squared_error(y_train_actual, train_predict_actual)
        test_mse = mean_squared_error(y_test_actual, test_predict_actual)
        train_mae = mean_absolute_error(y_train_actual, train_predict_actual)
        test_mae = mean_absolute_error(y_test_actual, test_predict_actual)
        train_r2 = r2_score(y_train_actual, train_predict_actual)
        test_r2 = r2_score(y_test_actual, test_predict_actual)
        
        logger.info(f"Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        logger.info(f"Testing MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        
        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        offset = sequence_length
        
        # Get the original dates for plotting
        train_dates = data.index[offset:offset + len(train_predict_actual)]
        test_dates = data.index[offset + len(train_predict_actual):offset + len(train_predict_actual) + len(test_predict_actual)]
        
        plt.plot(train_dates, y_train_actual, label='Training Actual')
        plt.plot(train_dates, train_predict_actual, label='Training Predicted')
        plt.plot(test_dates, y_test_actual, label='Testing Actual')
        plt.plot(test_dates, test_predict_actual, label='Testing Predicted')
        plt.title(f'LSTM Predictions vs Actual for {stock_symbol}')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'{stock_symbol}_lstm_predictions.png'))
        plt.close()
        
        # Save the Keras model separately (for easier loading)
        keras_model_path = os.path.join(model_dir, f"{stock_symbol}_lstm_model.h5")
        model.save(keras_model_path)
        
        # Create a dictionary of feature importance using SHAP
        # Note: SHAP for LSTM is more complex, so we'll use a simplified version
        feature_importance = calculate_lstm_feature_importance(model, X_test, feature_columns)
        
        # Create a complete model dict for saving
        model_dict = {
            'model': None,  # We don't save the Keras model directly with joblib
            'keras_model_path': keras_model_path,
            'sequence_length': sequence_length,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'scaler': scaler,
            'target_scaler': target_scaler,
            'performance': {
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2)
            },
            'feature_importance': feature_importance,
            'model_dir': model_dir,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save the model dict
        model_path = save_model(model_dict, stock_symbol, 'LSTM')
        if model_path:
            logger.info(f"LSTM model for {stock_symbol} saved to {model_path}")
            model_dict['model_path'] = model_path
        else:
            logger.error(f"Failed to save LSTM model for {stock_symbol}")
        
        return model_dict
    
    except Exception as e:
        logger.error(f"Error training LSTM model for {stock_symbol}: {str(e)}")
        return None

def calculate_lstm_feature_importance(model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """
    Calculate feature importance for LSTM models using a permutation importance approach.
    
    Args:
        model: Trained LSTM model
        X: Input sequences
        feature_names: Names of features
        
    Returns:
        Dictionary of feature importance scores
    """
    try:
        # Basic permutation importance
        # For each feature, we permute its values and measure the increase in error
        base_prediction = model.predict(X)
        base_error = np.mean((base_prediction.flatten() - X[:, -1, 0])**2)
        
        importance = {}
        for i in range(X.shape[2]):  # For each feature
            X_permuted = X.copy()
            # Permute the feature across all sequences and time steps
            for t in range(X.shape[1]):  # For each time step
                np.random.shuffle(X_permuted[:, t, i])
            
            # Make prediction with permuted feature
            perm_prediction = model.predict(X_permuted)
            perm_error = np.mean((perm_prediction.flatten() - X[:, -1, 0])**2)
            
            # Importance is the increase in error
            feature_importance = max(0, perm_error - base_error)
            importance[feature_names[i]] = float(feature_importance)
        
        # Normalize to sum to 1.0
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    except Exception as e:
        logger.error(f"Error calculating LSTM feature importance: {str(e)}")
        # Return equal importance if calculation fails
        return {feature: 1.0/len(feature_names) for feature in feature_names}

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
        # Load the model dict
        model_dict = load_model(model_path)
        if not model_dict:
            logger.error(f"Failed to load LSTM model from {model_path}")
            return None
        
        # Extract components
        keras_model_path = model_dict['keras_model_path']
        sequence_length = model_dict['sequence_length']
        feature_columns = model_dict['feature_columns']
        scaler = model_dict['scaler']
        target_scaler = model_dict['target_scaler']
        
        # Load the Keras model
        model = keras_load_model(keras_model_path)
        
        # Verify that all required features are present
        missing_features = [f for f in feature_columns if f not in latest_data.columns]
        if missing_features:
            logger.error(f"Missing features for prediction: {missing_features}")
            return None
        
        # Prepare the sequence
        # We need the last 'sequence_length' data points
        if len(latest_data) < sequence_length:
            logger.error(f"Not enough data points. Need at least {sequence_length}")
            return None
        
        # Extract and scale the data
        input_data = latest_data[feature_columns].values[-sequence_length:]
        scaled_data = scaler.transform(input_data)
        
        # Reshape to match LSTM input shape (1, sequence_length, n_features)
        X = scaled_data.reshape(1, sequence_length, len(feature_columns))
        
        # Make prediction
        scaled_prediction = model.predict(X)[0, 0]
        
        # Inverse transform to get the actual price
        # We need to create a dummy array with the right shape
        dummy = np.zeros((1, len(feature_columns)))
        dummy[0, 0] = scaled_prediction
        prediction = scaler.inverse_transform(dummy)[0, 0]
        
        return float(prediction)
    
    except Exception as e:
        logger.error(f"Error predicting with LSTM model: {str(e)}")
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
        # Load the model dict
        model_dict = load_model(model_path)
        if not model_dict:
            logger.error(f"Failed to load LSTM model from {model_path}")
            return None
        
        # Extract components
        keras_model_path = model_dict['keras_model_path']
        sequence_length = model_dict['sequence_length']
        feature_columns = model_dict['feature_columns']
        target_column = model_dict['target_column']
        scaler = model_dict['scaler']
        target_scaler = model_dict['target_scaler']
        
        # Load the Keras model
        model = keras_load_model(keras_model_path)
        
        # Verify that all required features are present
        missing_features = [f for f in feature_columns if f not in latest_data.columns]
        if missing_features:
            logger.error(f"Missing features for prediction: {missing_features}")
            return None
        
        # Start with the last 'sequence_length' points from the data
        if len(latest_data) < sequence_length:
            logger.error(f"Not enough data points. Need at least {sequence_length}")
            return None
        
        # Make a copy of the latest data to avoid modifying the original
        current_data = latest_data.copy()
        predictions = []
        
        for _ in range(days):
            # Get the last sequence_length data points
            input_data = current_data[feature_columns].values[-sequence_length:]
            scaled_data = scaler.transform(input_data)
            
            # Reshape for prediction
            X = scaled_data.reshape(1, sequence_length, len(feature_columns))
            
            # Make prediction
            scaled_prediction = model.predict(X)[0, 0]
            
            # Inverse transform
            dummy = np.zeros((1, len(feature_columns)))
            dummy[0, 0] = scaled_prediction
            prediction = scaler.inverse_transform(dummy)[0, 0]
            predictions.append(float(prediction))
            
            # Create a new row for the next prediction
            new_row = current_data.iloc[-1:].copy()
            new_row.index = [new_row.index[0] + timedelta(days=1)]
            
            # Update with prediction
            new_row[target_column] = prediction
            
            # For other features, use the last known values (simplification)
            # In a real-world scenario, we might have more sophisticated ways to project other features
            
            # Append to current data
            current_data = pd.concat([current_data, new_row])
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error making future predictions with LSTM model: {str(e)}")
        return None
