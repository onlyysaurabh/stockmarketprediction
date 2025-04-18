import numpy as np
import pandas as pd
import logging
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import shap
from ._base_model import save_model, load_model, prepare_data_for_model

logger = logging.getLogger(__name__)

def train_svm_model(stock_symbol: str, data: pd.DataFrame, target_column: str = None, 
                   feature_columns: List[str] = None, test_size: float = 0.2) -> Dict:
    """
    Train an SVM regression model for stock price prediction.
    
    Args:
        stock_symbol: Stock symbol for the model
        data: DataFrame with features and target
        target_column: Column to predict (default: stock's Close price)
        feature_columns: List of feature column names (if None, uses all except target)
        test_size: Proportion of data to use for testing
        
    Returns:
        Dict with trained model, feature importance, and performance metrics
    """
    # Set default target column if not provided
    if target_column is None:
        target_column = f"{stock_symbol}_Close"
    
    # Set default feature columns if not provided
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]
        # Remove any columns that might cause data leakage
        feature_columns = [col for col in feature_columns if not (col.startswith(stock_symbol) and 
                                                                 ('Close' in col or 'Open' in col))]
    
    logger.info(f"Training SVM model for {stock_symbol} with {len(feature_columns)} features.")
    
    # Prepare data for model training
    X, y = prepare_data_for_model(data, target_column, feature_columns)
    if X is None or y is None:
        logger.error(f"Data preparation failed for {stock_symbol}.")
        return None
    
    # Create training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter optimization using grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    
    logger.info("Performing grid search for SVM hyperparameters...")
    grid_search = GridSearchCV(
        SVR(),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model from grid search
    best_model = grid_search.best_estimator_
    logger.info(f"Best SVM parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    # Evaluate model
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    logger.info(f"Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    logger.info(f"Testing MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Calculate feature importance using SHAP
    shap_values, feature_importance = calculate_svm_shap_importance(best_model, X_test_scaled, X.columns)
    
    # Create a complete model dict with scaler for future predictions
    model_dict = {
        'model': best_model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'target_column': target_column,
        'performance': {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2)
        },
        'feature_importance': feature_importance,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save the model
    model_path = save_model(model_dict, stock_symbol, 'SVM')
    if model_path:
        logger.info(f"SVM model for {stock_symbol} saved to {model_path}")
        model_dict['model_path'] = model_path
    else:
        logger.error(f"Failed to save SVM model for {stock_symbol}")
    
    return model_dict

def calculate_svm_shap_importance(model, X_scaled, feature_names) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Calculate feature importance for SVM models using SHAP.
    
    Args:
        model: Trained SVM model
        X_scaled: Scaled feature matrix 
        feature_names: Names of features
        
    Returns:
        Tuple of (shap_values, feature_importance_dict)
    """
    try:
        # Create a SHAP explainer
        explainer = shap.KernelExplainer(model.predict, X_scaled[:100])  # Using a subset for performance
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_scaled[:100])
        
        # Calculate mean absolute SHAP value per feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create dictionary of feature importance
        feature_importance = {}
        for i, feature in enumerate(feature_names):
            feature_importance[feature] = float(mean_shap[i])
        
        # Normalize to sum to 1.0
        total = sum(feature_importance.values())
        if total > 0:
            feature_importance = {k: v/total for k, v in feature_importance.items()}
        
        return shap_values, feature_importance
    
    except Exception as e:
        logger.error(f"Error calculating SHAP values for SVM model: {str(e)}")
        # Return empty results if SHAP calculation fails
        return np.array([]), {feature: 0.0 for feature in feature_names}

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
        # Load the model
        model_dict = load_model(model_path)
        if not model_dict:
            logger.error(f"Failed to load SVM model from {model_path}")
            return None
        
        # Extract components from the model dict
        model = model_dict['model']
        scaler = model_dict['scaler']
        feature_columns = model_dict['feature_columns']
        
        # Verify that all required features are present
        missing_features = [f for f in feature_columns if f not in latest_data.columns]
        if missing_features:
            logger.error(f"Missing features for prediction: {missing_features}")
            return None
        
        # Prepare features for prediction
        X = latest_data[feature_columns].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        return float(prediction)
    
    except Exception as e:
        logger.error(f"Error predicting with SVM model: {str(e)}")
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
            logger.error(f"Failed to load SVM model from {model_path}")
            return None
        
        # Extract components
        model = model_dict['model']
        scaler = model_dict['scaler']
        feature_columns = model_dict['feature_columns']
        target_column = model_dict['target_column']
        
        predictions = []
        current_data = latest_data.copy()
        
        for _ in range(days):
            # Prepare features for prediction
            X = current_data[feature_columns].iloc[-1:].values
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = float(model.predict(X_scaled)[0])
            predictions.append(prediction)
            
            # Update data with the new prediction for next iteration
            # Create a new row based on the last row
            new_row = current_data.iloc[-1:].copy()
            
            # Set the index to be the next day
            new_row.index = [new_row.index[0] + timedelta(days=1)]
            
            # Update the target column with the prediction
            new_row[target_column] = prediction
            
            # Append to current_data
            current_data = pd.concat([current_data, new_row])
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error making future predictions with SVM model: {str(e)}")
        return None
