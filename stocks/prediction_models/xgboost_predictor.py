import numpy as np
import pandas as pd
import logging
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import shap
from ._base_model import save_model, load_model, prepare_data_for_model

logger = logging.getLogger(__name__)

def train_xgboost_model(stock_symbol: str, data: pd.DataFrame, target_column: str = None,
                       feature_columns: List[str] = None, test_size: float = 0.2) -> Dict:
    """
    Train an XGBoost model for stock price prediction.
    
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
    
    logger.info(f"Training XGBoost model for {stock_symbol} with {len(feature_columns)} features.")
    
    # Prepare data for model training
    X, y = prepare_data_for_model(data, target_column, feature_columns)
    if X is None or y is None:
        logger.error(f"Data preparation failed for {stock_symbol}.")
        return None
    
    # Create training, validation, and testing datasets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size*1.5, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, shuffle=False)
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection using a base XGBoost model
    base_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1)
    base_model.fit(X_train_scaled, y_train)
    
    # Select important features
    selection = SelectFromModel(base_model, threshold="median", prefit=True)
    X_train_selected = selection.transform(X_train_scaled)
    X_val_selected = selection.transform(X_val_scaled)
    X_test_selected = selection.transform(X_test_scaled)
    
    # Get selected feature indices
    selected_indices = selection.get_support()
    selected_features = [feature for i, feature in enumerate(X.columns) if selected_indices[i]]
    logger.info(f"Selected {len(selected_features)} features after feature selection")
    
    # Enhanced hyperparameter optimization using randomized search
    param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [3, 5, 7, 9, 12],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [0, 0.1, 1, 10]
    }
    
    logger.info("Performing randomized search for XGBoost hyperparameters...")
    random_search = RandomizedSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1),
        param_distributions=param_grid,
        n_iter=30,  # Increased from 20
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Use selected features for hyperparameter tuning
    random_search.fit(X_train_selected, y_train)
    
    # Get best model from random search
    best_params = random_search.best_params_
    logger.info(f"Best XGBoost parameters: {best_params}")
    
    # Train final model with early stopping using validation set
    final_model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', n_jobs=-1)
    final_model.fit(
        X_train_selected, y_train,
        eval_set=[(X_val_selected, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Save the early stopping rounds
    best_iteration = getattr(final_model, "best_iteration", None) or len(final_model.evals_result()["validation_0"]["rmse"])
    
    # Make predictions
    y_train_pred = final_model.predict(X_train_selected)
    y_val_pred = final_model.predict(X_val_selected)
    y_test_pred = final_model.predict(X_test_selected)
    
    # Evaluate model on all datasets
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    logger.info(f"Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    logger.info(f"Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    logger.info(f"Testing MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Calculate feature importance (both built-in and SHAP)
    # We need to map back to the original feature space
    feature_map = {i: feature for i, feature in enumerate(selected_features)}
    native_importance = get_xgboost_feature_importance(final_model, selected_features)
    shap_importance = calculate_xgboost_shap_importance(final_model, X_test_selected, selected_features)
    
    # Combine both importance methods in a single dictionary
    feature_importance = {
        'native': native_importance,
        'shap': shap_importance
    }
    
    # Create a complete model dict for future predictions
    model_dict = {
        'model': final_model,
        'feature_selector': selection,
        'scaler': scaler,
        'original_features': feature_columns,
        'selected_features': selected_features,
        'target_column': target_column,
        'best_iteration': best_iteration,
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
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save the model
    model_path = save_model(model_dict, stock_symbol, 'XGBOOST')
    if model_path:
        logger.info(f"XGBoost model for {stock_symbol} saved to {model_path}")
        model_dict['model_path'] = model_path
    else:
        logger.error(f"Failed to save XGBoost model for {stock_symbol}")
    
    return model_dict

def get_xgboost_feature_importance(model, feature_names) -> Dict[str, float]:
    """
    Get feature importance from XGBoost's built-in feature_importances_ attribute.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        Dictionary of feature importance scores
    """
    try:
        # Get feature importance from the model, trying different methods
        try:
            # Get feature importance by gain (improvement in accuracy brought by a feature)
            importance = model.get_booster().get_score(importance_type='gain')
            
            # Convert to dictionary with proper feature names
            importance_dict = {}
            for f_name, score in importance.items():
                # XGBoost feature names might be f0, f1, etc.
                if f_name.startswith('f'):
                    try:
                        idx = int(f_name[1:])
                        if idx < len(feature_names):
                            importance_dict[feature_names[idx]] = float(score)
                    except ValueError:
                        importance_dict[f_name] = float(score)
                else:
                    importance_dict[f_name] = float(score)
        except:
            # Fallback to feature_importances_ attribute
            importance = model.feature_importances_
            importance_dict = {feature: float(imp) for feature, imp in zip(feature_names, importance)}
        
        # Normalize to sum to 1.0
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        return importance_dict
    
    except Exception as e:
        logger.error(f"Error getting XGBoost feature importance: {str(e)}")
        # Return empty dictionary if calculation fails
        return {feature: 0.0 for feature in feature_names}

def calculate_xgboost_shap_importance(model, X, feature_names) -> Dict[str, float]:
    """
    Calculate feature importance for XGBoost using SHAP.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        feature_names: Names of features
        
    Returns:
        Dict of feature importance scores
    """
    try:
        # Calculate SHAP values using TreeExplainer which is optimized for tree-based models
        explainer = shap.TreeExplainer(model)
        
        # Use a sample of X if it's large (for performance)
        sample_size = min(len(X), 500)  # Limit to 500 samples for efficiency
        if sample_size < len(X):
            # Use random sampling without replacement
            sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
            X_sample = X[sample_indices]
        else:
            X_sample = X
            
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP value per feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create dictionary of feature importance
        importance_dict = {}
        for i, feature in enumerate(feature_names):
            importance_dict[feature] = float(mean_shap[i])
        
        # Normalize to sum to 1.0
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        return importance_dict
    
    except Exception as e:
        logger.error(f"Error calculating SHAP values for XGBoost model: {str(e)}")
        # Return empty dictionary if calculation fails
        return {feature: 0.0 for feature in feature_names}

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
            logger.error(f"Failed to load XGBoost model from {model_path}")
            return None
        
        # Extract components from the model dict
        model = model_dict['model']
        feature_selector = model_dict.get('feature_selector')
        scaler = model_dict.get('scaler')
        original_features = model_dict.get('original_features', [])
        selected_features = model_dict.get('selected_features', [])
        
        # If we have original features but no feature_selector, we're using an old model
        if not feature_selector and not selected_features and original_features:
            feature_columns = original_features
            
            # Verify that all required features are present
            missing_features = [f for f in feature_columns if f not in latest_data.columns]
            if missing_features:
                logger.error(f"Missing features for prediction: {missing_features}")
                return None
            
            # Prepare features for prediction using old method
            X = latest_data[feature_columns].values.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            return float(prediction)
        
        # Using the improved model with feature selection
        # Verify that all required original features are present
        missing_features = [f for f in original_features if f not in latest_data.columns]
        if missing_features:
            logger.error(f"Missing features for prediction: {missing_features}")
            return None
            
        # Prepare features for prediction
        X = latest_data[original_features].values.reshape(1, -1)
        
        # Scale features if scaler is available
        if scaler:
            X = scaler.transform(X)
            
        # Apply feature selection if available
        if feature_selector:
            X = feature_selector.transform(X)
        elif selected_features and len(selected_features) < len(original_features):
            # Manual selection if selected_features are available but no feature_selector
            selected_indices = [i for i, feat in enumerate(original_features) if feat in selected_features]
            X = X[:, selected_indices]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return float(prediction)
    
    except Exception as e:
        logger.error(f"Error predicting with XGBoost model: {str(e)}")
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
            logger.error(f"Failed to load XGBoost model from {model_path}")
            return None
        
        # Extract components
        model = model_dict['model']
        feature_selector = model_dict.get('feature_selector')
        scaler = model_dict.get('scaler')
        original_features = model_dict.get('original_features', [])
        selected_features = model_dict.get('selected_features', [])
        target_column = model_dict['target_column']
        
        predictions = []
        current_data = latest_data.copy()
        
        for _ in range(days):
            # Prepare features for prediction
            X = current_data[original_features].iloc[-1:].values
            
            # Scale features if scaler is available
            if scaler:
                X = scaler.transform(X)
                
            # Apply feature selection if available
            if feature_selector:
                X = feature_selector.transform(X)
            elif selected_features and len(selected_features) < len(original_features):
                # Manual selection if selected_features are available but no feature_selector
                selected_indices = [i for i, feat in enumerate(original_features) if feat in selected_features]
                X = X[:, selected_indices]
            
            # Make prediction
            prediction = float(model.predict(X)[0])
            predictions.append(prediction)
            
            # Create a new row for the next prediction
            new_row = current_data.iloc[-1:].copy()
            
            # Set the index to be the next day
            new_row.index = [new_row.index[0] + timedelta(days=1)]
            
            # Update the target column with the prediction
            new_row[target_column] = prediction
            
            # Append to current_data
            current_data = pd.concat([current_data, new_row])
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error making future predictions with XGBoost model: {str(e)}")
        return None
