import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from ._base_model import save_model, load_model, MODEL_STORAGE_DIR

# Suppress TensorFlow INFO and WARNING logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

# Configuration options - can be optimized per stock if needed
DEFAULT_SEQUENCE_LENGTH = 40  # Default window of historical data to use
TARGET_COLUMN = 'Target_Close'  # Next day's closing price

def create_lstm_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of data for LSTM training.
    
    Args:
        data (np.ndarray): Scaled data array (features + target)
        sequence_length (int): The number of time steps in each sequence
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (sequences of features) and y (target values)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Sequence: data from i to i + sequence_length
        seq = data[i:(i + sequence_length)]
        # Target: the target value at the end of the sequence
        target = data[i + sequence_length, 0]  # Target assumed to be first column
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, complexity='medium', dropout_rate=0.3, regularization_factor=0.01):
    """
    Build an LSTM model with customizable architecture complexity.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        complexity: Model complexity - 'simple', 'medium', or 'complex'
        dropout_rate: Rate of dropout for regularization
        regularization_factor: L1L2 regularization factor
        
    Returns:
        Compiled Keras model
    """
    reg = l1_l2(l1=regularization_factor, l2=regularization_factor)
    
    if complexity == 'simple':
        model = Sequential([
            LSTM(units=50, return_sequences=False, input_shape=input_shape, 
                 kernel_regularizer=reg, recurrent_regularizer=reg),
            Dropout(dropout_rate),
            Dense(units=1)
        ])
    elif complexity == 'medium':
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape, 
                 kernel_regularizer=reg, recurrent_regularizer=reg),
            Dropout(dropout_rate),
            LSTM(units=64, return_sequences=False, 
                 kernel_regularizer=reg, recurrent_regularizer=reg),
            Dropout(dropout_rate),
            Dense(units=32, activation='relu'),
            Dense(units=1)
        ])
    else:  # complex
        model = Sequential([
            Bidirectional(LSTM(units=64, return_sequences=True, input_shape=input_shape, 
                              kernel_regularizer=reg, recurrent_regularizer=reg)),
            Dropout(dropout_rate),
            BatchNormalization(),
            Bidirectional(LSTM(units=64, return_sequences=False, 
                              kernel_regularizer=reg, recurrent_regularizer=reg)),
            Dropout(dropout_rate),
            BatchNormalization(),
            Dense(units=32, activation='relu'),
            Dense(units=1)
        ])
    
    # Use Huber loss for robustness to outliers
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
    return model

def build_attention_lstm_model(input_shape, units=64, dropout_rate=0.2, regularization_factor=0.01):
    """
    Build an LSTM model with attention mechanism.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        units: Number of units in LSTM layers
        dropout_rate: Rate of dropout for regularization
        regularization_factor: L1L2 regularization factor
        
    Returns:
        Compiled Keras model
    """
    reg = l1_l2(l1=regularization_factor, l2=regularization_factor)
    
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # LSTM layer
    lstm_out = LSTM(units=units, return_sequences=True, 
                   kernel_regularizer=reg, recurrent_regularizer=reg)(inputs)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    # Multi-head attention layer
    attention_output = MultiHeadAttention(num_heads=4, key_dim=units//4)(lstm_out, lstm_out)
    attention_output = Dropout(dropout_rate)(attention_output)
    
    # Add & Normalize
    normalized = BatchNormalization()(attention_output + lstm_out)
    
    # Another LSTM layer
    lstm_out2 = LSTM(units=units, return_sequences=False)(normalized)
    lstm_out2 = Dropout(dropout_rate)(lstm_out2)
    
    # Dense layers
    dense = Dense(units=32, activation='relu')(lstm_out2)
    output = Dense(units=1)(dense)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    # Use Huber loss for robustness to outliers
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
    return model

def train_lstm_model(stock_symbol: str, data: pd.DataFrame, target_column: str = None,
                   feature_columns: List[str] = None, test_size: float = 0.2,
                   val_size: float = 0.1, sequence_length: int = None) -> Dict:
    """
    Train an LSTM model for stock price prediction with improved architecture and training.
    
    Args:
        stock_symbol: Stock symbol for the model
        data: DataFrame with features and target, must be indexed by date
        target_column: Column to predict (default: stock's Close price)
        feature_columns: List of feature column names to include
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        sequence_length: Number of time steps to use for each prediction
        
    Returns:
        Dict with trained model, scalers, and performance metrics
    """
    try:
        # Set default target column if not provided
        if target_column is None:
            target_column = f"{stock_symbol}_Close"
        
        # Set default sequence length if not provided
        if sequence_length is None:
            sequence_length = DEFAULT_SEQUENCE_LENGTH
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Set default feature columns if not provided
        if feature_columns is None:
            # For LSTM, include relevant price and volume data
            feature_columns = [col for col in data.columns 
                              if any(price_type in col for price_type in 
                                   ['Open', 'High', 'Low', 'Close', 'Volume'])]
            
            # Also include sentiment features if available
            sentiment_features = ['avg_positive', 'avg_negative', 'avg_neutral', 'news_count']
            for feat in sentiment_features:
                if feat in data.columns:
                    feature_columns.append(feat)
        
        logger.info(f"Training LSTM model for {stock_symbol} with {len(feature_columns)} features")
        
        # Check if target column exists in features, if not, add it
        if target_column not in feature_columns:
            feature_columns.append(target_column)
        
        # Extract features and ensure target column is the first column
        feature_columns.remove(target_column)
        selected_columns = [target_column] + feature_columns  # Target is first for easier sequence creation
        
        # Extract selected features
        dataset = data[selected_columns].values
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Create a separate scaler for the target column for easy inverse transform
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_data = data[[target_column]].values
        target_scaler.fit(target_data)
        
        # Create sequences
        X, y = create_lstm_sequences(scaled_data, sequence_length)
        
        if len(X) < 100:  # Minimum data required
            logger.error(f"Not enough data for LSTM training. Need at least {100+sequence_length} data points.")
            return None
        
        # Split into training, validation and testing sets
        # First split out test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Keep chronological order
        )
        
        # Then split training set into training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size/(1-test_size), shuffle=False
        )
        
        # Create a directory for model checkpoints and diagnostics
        model_dir = os.path.join(MODEL_STORAGE_DIR, 'keras_models', stock_symbol)
        os.makedirs(model_dir, exist_ok=True)
        
        # Try different model architectures (simple, medium, complex) based on data size
        if len(X_train) < 500:
            model_complexity = 'simple'
            batch_size = 16
            epochs = 150
        elif len(X_train) < 2000:
            model_complexity = 'medium'
            batch_size = 32
            epochs = 200
        else:
            model_complexity = 'complex'
            batch_size = 64
            epochs = 300
        
        logger.info(f"Using {model_complexity} LSTM architecture with {len(X_train)} training samples")
        
        # Define callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, f"{stock_symbol}_lstm_checkpoint.h5"),
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Build and train the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Try both standard LSTM and attention LSTM models
        models_to_try = {
            'standard': build_lstm_model(input_shape, complexity=model_complexity),
            'attention': build_attention_lstm_model(input_shape)
        }
        
        best_val_loss = float('inf')
        best_model_type = None
        best_model = None
        best_history = None
        
        # Train each model type and select the best one
        for model_type, model in models_to_try.items():
            logger.info(f"Training {model_type} LSTM model...")
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Check if this model is better than previous ones
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_type = model_type
                best_model = model
                best_history = history
        
        logger.info(f"Selected {best_model_type} LSTM model with validation loss: {best_val_loss:.6f}")
        
        # Save training history plot
        plt.figure(figsize=(12, 6))
        plt.plot(best_history.history['loss'], label='Training Loss')
        plt.plot(best_history.history['val_loss'], label='Validation Loss')
        plt.title(f'LSTM Training History for {stock_symbol} ({best_model_type})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (Huber)')
        plt.legend()
        plt.savefig(os.path.join(model_dir, f'{stock_symbol}_lstm_training.png'))
        plt.close()
        
        # Make predictions
        train_predict = best_model.predict(X_train)
        val_predict = best_model.predict(X_val)
        test_predict = best_model.predict(X_test)
        
        # Inverse transform to get actual prices
        # Create dummy arrays with the right shape for inverse_transform
        train_predict_dummy = np.zeros((len(train_predict), len(selected_columns)))
        train_predict_dummy[:, 0] = train_predict.flatten()
        train_predict_actual = scaler.inverse_transform(train_predict_dummy)[:, 0]
        
        val_predict_dummy = np.zeros((len(val_predict), len(selected_columns)))
        val_predict_dummy[:, 0] = val_predict.flatten()
        val_predict_actual = scaler.inverse_transform(val_predict_dummy)[:, 0]
        
        test_predict_dummy = np.zeros((len(test_predict), len(selected_columns)))
        test_predict_dummy[:, 0] = test_predict.flatten()
        test_predict_actual = scaler.inverse_transform(test_predict_dummy)[:, 0]
        
        # Get actual target values
        y_train_dummy = np.zeros((len(y_train), len(selected_columns)))
        y_train_dummy[:, 0] = y_train
        y_train_actual = scaler.inverse_transform(y_train_dummy)[:, 0]
        
        y_val_dummy = np.zeros((len(y_val), len(selected_columns)))
        y_val_dummy[:, 0] = y_val
        y_val_actual = scaler.inverse_transform(y_val_dummy)[:, 0]
        
        y_test_dummy = np.zeros((len(y_test), len(selected_columns)))
        y_test_dummy[:, 0] = y_test
        y_test_actual = scaler.inverse_transform(y_test_dummy)[:, 0]
        
        # Calculate performance metrics
        train_mse = mean_squared_error(y_train_actual, train_predict_actual)
        val_mse = mean_squared_error(y_val_actual, val_predict_actual)
        test_mse = mean_squared_error(y_test_actual, test_predict_actual)
        
        train_mae = mean_absolute_error(y_train_actual, train_predict_actual)
        val_mae = mean_absolute_error(y_val_actual, val_predict_actual)
        test_mae = mean_absolute_error(y_test_actual, test_predict_actual)
        
        train_r2 = r2_score(y_train_actual, train_predict_actual)
        val_r2 = r2_score(y_val_actual, val_predict_actual)
        test_r2 = r2_score(y_test_actual, test_predict_actual)
        
        logger.info(f"Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        logger.info(f"Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        logger.info(f"Testing MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        
        # Save the Keras model separately (for easier loading)
        keras_model_path = os.path.join(model_dir, f"{stock_symbol}_lstm_model.h5")
        best_model.save(keras_model_path)
        
        # Calculate feature importance using permutation importance
        feature_importance = calculate_lstm_feature_importance(best_model, X_test, selected_columns)
        
        # Plot predictions vs actual
        # Get dates aligned with predictions
        offset = sequence_length
        train_offset = offset
        val_offset = train_offset + len(train_predict_actual)
        test_offset = val_offset + len(val_predict_actual)
        
        # Get the original dates for plotting
        train_dates = data.index[train_offset:train_offset + len(train_predict_actual)]
        val_dates = data.index[val_offset:val_offset + len(val_predict_actual)]
        test_dates = data.index[test_offset:test_offset + len(test_predict_actual)]
        
        plt.figure(figsize=(12, 6))
        plt.plot(train_dates, y_train_actual, label='Training Actual', alpha=0.7)
        plt.plot(train_dates, train_predict_actual, label='Training Predicted', alpha=0.7)
        plt.plot(val_dates, y_val_actual, label='Validation Actual', alpha=0.7)
        plt.plot(val_dates, val_predict_actual, label='Validation Predicted', alpha=0.7)
        plt.plot(test_dates, y_test_actual, label='Testing Actual')
        plt.plot(test_dates, test_predict_actual, label='Testing Predicted')
        plt.title(f'LSTM Predictions vs Actual for {stock_symbol}')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'{stock_symbol}_lstm_predictions.png'))
        plt.close()
        
        # Create a complete model dict for saving
        model_dict = {
            'model': None,  # We don't save the Keras model directly with joblib
            'keras_model_path': keras_model_path,
            'model_type': best_model_type,
            'sequence_length': sequence_length,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'selected_columns': selected_columns,  # All columns including target
            'scaler': scaler,
            'target_scaler': target_scaler,
            'performance': {
                'train_mse': float(train_mse),
                'val_mse': float(val_mse),
                'test_mse': float(test_mse),
                'train_mae': float(train_mae),
                'val_mae': float(val_mae),
                'test_mae': float(test_mae),
                'train_r2': float(train_r2),
                'val_r2': float(val_r2),
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
        logger.error(f"Error training LSTM model for {stock_symbol}: {str(e)}", exc_info=True)
        return None

def calculate_lstm_feature_importance(model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """
    Calculate feature importance for LSTM models using permutation importance.
    
    Args:
        model: Trained LSTM model
        X: Input sequences
        feature_names: Names of features
        
    Returns:
        Dictionary of feature importance scores
    """
    try:
        # Basic permutation importance approach
        # For each feature, we permute its values and measure the increase in error
        
        # First, get baseline predictions
        baseline_preds = model.predict(X)
        baseline_error = np.mean(np.square(baseline_preds.flatten() - X[:, -1, 0]))
        
        importance = {}
        
        # For each feature, permute its values and calculate importance
        for i in range(X.shape[2]):  # For each feature
            # Make a copy of the input data
            X_permuted = X.copy()
            
            # Permute the feature across all sequences and time steps
            for t in range(X.shape[1]):  # For each time step
                np.random.shuffle(X_permuted[:, t, i])
            
            # Make predictions with permuted feature
            perm_preds = model.predict(X_permuted)
            perm_error = np.mean(np.square(perm_preds.flatten() - X[:, -1, 0]))
            
            # Importance is the increase in error (can't be negative)
            feature_importance = max(0, perm_error - baseline_error)
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
    Make a prediction for the next day's stock price.
    
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
        selected_columns = model_dict.get('selected_columns')
        feature_columns = model_dict['feature_columns']
        target_column = model_dict['target_column']
        scaler = model_dict['scaler']
        
        # Handle case where we don't have selected_columns (older models)
        if selected_columns is None:
            selected_columns = [target_column] + feature_columns
        
        # Load the Keras model
        model = keras_load_model(keras_model_path)
        
        # Verify that all required features are present
        missing_features = [f for f in feature_columns if f not in latest_data.columns]
        if missing_features:
            logger.error(f"Missing features for prediction: {missing_features}")
            return None
        
        # Check if we have enough data for the sequence
        if len(latest_data) < sequence_length:
            logger.error(f"Not enough data for prediction. Need at least {sequence_length} samples.")
            return None
        
        # Prepare input data - need both features and target in same order as training
        input_df = latest_data[feature_columns].copy()
        
        # Add target column (value doesn't matter, will be scaled and discarded)
        if target_column not in input_df.columns:
            input_df[target_column] = input_df[feature_columns[0]]  # Placeholder
        
        # Reorder columns to match training data
        input_df = input_df[[target_column] + feature_columns]
        
        # Get the last sequence_length data points
        input_data = input_df.values[-sequence_length:]
        
        # Scale the data
        scaled_data = scaler.transform(input_data)
        
        # Reshape to match LSTM input shape [1, sequence_length, features]
        X = scaled_data.reshape(1, sequence_length, len(selected_columns))
        
        # Make prediction
        scaled_prediction = model.predict(X)[0, 0]
        
        # Inverse transform to get actual price
        # Create dummy array with the right shape
        dummy = np.zeros((1, len(selected_columns)))
        dummy[0, 0] = scaled_prediction
        prediction = scaler.inverse_transform(dummy)[0, 0]
        
        return float(prediction)
    
    except Exception as e:
        logger.error(f"Error predicting with LSTM model: {str(e)}", exc_info=True)
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
        selected_columns = model_dict.get('selected_columns')
        feature_columns = model_dict['feature_columns']
        target_column = model_dict['target_column']
        scaler = model_dict['scaler']
        
        # Handle case where we don't have selected_columns (older models)
        if selected_columns is None:
            selected_columns = [target_column] + feature_columns
        
        # Load the Keras model
        model = keras_load_model(keras_model_path)
        
        # Verify required features
        missing_features = [f for f in feature_columns if f not in latest_data.columns]
        if missing_features:
            logger.error(f"Missing features for prediction: {missing_features}")
            return None
        
        # Check if we have enough data
        if len(latest_data) < sequence_length:
            logger.error(f"Not enough data for prediction. Need at least {sequence_length} samples.")
            return None
        
        # Make a copy of the data to extend with predictions
        working_df = latest_data.copy()
        predictions = []
        
        for _ in range(days):
            # Prepare input data
            input_df = working_df[feature_columns].copy()
            
            # Add target column (value doesn't matter for input)
            if target_column not in input_df.columns:
                input_df[target_column] = input_df[feature_columns[0]]  # Placeholder
            
            # Reorder columns
            input_df = input_df[[target_column] + feature_columns]
            
            # Get latest sequence
            input_data = input_df.values[-sequence_length:]
            
            # Scale the data
            scaled_data = scaler.transform(input_data)
            
            # Reshape for LSTM
            X = scaled_data.reshape(1, sequence_length, len(selected_columns))
            
            # Make prediction
            scaled_prediction = model.predict(X)[0, 0]
            
            # Inverse transform
            dummy = np.zeros((1, len(selected_columns)))
            dummy[0, 0] = scaled_prediction
            prediction = float(scaler.inverse_transform(dummy)[0, 0])
            predictions.append(prediction)
            
            # Create a new row with prediction
            new_row = pd.DataFrame(index=[working_df.index[-1] + timedelta(days=1)])
            
            # For features, use the last known values (simplistic approach)
            for col in feature_columns:
                new_row[col] = working_df[col].iloc[-1]
            
            # Update target column with prediction
            new_row[target_column] = prediction
            
            # Add to working data
            working_df = pd.concat([working_df, new_row])
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error making future predictions with LSTM model: {str(e)}", exc_info=True)
        return None
