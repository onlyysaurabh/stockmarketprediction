import joblib
import os
import logging
from typing import Any, Optional, Dict, List # Removed Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# Define a directory to store trained models
# Consider making this configurable via Django settings
MODEL_STORAGE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models')
os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)

def save_model(model: Any, stock_symbol: str, model_type: str) -> Optional[str]:
    """
    Saves a trained model object to a file.

    Args:
        model: The trained model object (e.g., scikit-learn estimator, XGBoost model).
        stock_symbol (str): The stock symbol the model is for.
        model_type (str): The type of the model (e.g., 'SVM', 'XGBOOST').

    Returns:
        Optional[str]: The absolute path to the saved model file, or None if saving failed.
    """
    try:
        # Create a directory for this stock if it doesn't exist
        stock_dir = os.path.join(MODEL_STORAGE_DIR, stock_symbol)
        os.makedirs(stock_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stock_symbol}_{model_type}_{timestamp}.joblib"
        filepath = os.path.join(stock_dir, filename)

        # Use joblib for efficient saving of scikit-learn models and others
        joblib.dump(model, filepath)
        logger.info(f"Model for {stock_symbol} ({model_type}) saved successfully to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving model for {stock_symbol} ({model_type}): {e}", exc_info=True)
        return None

def load_model(filepath: str) -> Optional[Any]:
    """
    Loads a model object from a file.

    Args:
        filepath (str): The absolute path to the saved model file.

    Returns:
        Optional[Any]: The loaded model object, or None if loading failed.
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Model file not found at path: {filepath}")
            return None

        model = joblib.load(filepath)
        logger.info(f"Model loaded successfully from: {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {e}", exc_info=True)
        return None

def prepare_data_for_model(data: pd.DataFrame, target_column: str, features: List[str]):
    """
    Basic data preparation: select features and target, handle NaNs.
    Specific models might require more advanced preprocessing (scaling, differencing etc.)
    which should be handled in their respective files or a more complex pipeline.

    Args:
        data (pd.DataFrame): Input DataFrame from prediction_data_service.
        target_column (str): The name of the column to predict.
        features (List[str]): List of feature column names.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X (features) and y (target) DataFrames/Series.
                                         Returns (None, None) if columns are missing.
    """
    if target_column not in data.columns:
        logger.error(f"Target column '{target_column}' not found in DataFrame.")
        return None, None
    
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        logger.error(f"Feature columns not found in DataFrame: {missing_features}")
        return None, None

    # Ensure no NaNs in features or target before splitting
    data_subset = data[features + [target_column]].copy()
    initial_rows = len(data_subset)
    data_subset.dropna(inplace=True)
    if len(data_subset) < initial_rows:
         logger.warning(f"Dropped {initial_rows - len(data_subset)} rows with NaNs during final data prep.")

    if data_subset.empty:
        logger.error("DataFrame is empty after selecting features/target and dropping NaNs.")
        return None, None

    X = data_subset[features]
    y = data_subset[target_column]

    return X, y


# Placeholder for feature importance calculation - specific implementations needed per model type
def calculate_feature_importance(model: Any, model_type: str, X: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Calculates feature importance based on the model type.
    Requires specific implementation for each model (e.g., SHAP, built-in).

    Args:
        model: The trained model object.
        model_type (str): Type of the model ('XGBOOST', 'SVM', etc.).
        X (pd.DataFrame): The feature data used for training (needed for SHAP).

    Returns:
        Optional[Dict[str, float]]: Dictionary of feature names and their importance scores.
    """
    logger.warning(f"Feature importance calculation not yet implemented for model type: {model_type}")
    # TODO: Implement SHAP for SVM, LSTM?
    # TODO: Implement native importance for XGBoost
    # TODO: SARIMA importance might be based on coefficients or other methods
    if model_type == 'XGBOOST':
        try:
            # XGBoost has built-in feature importance
            importance = model.get_score(importance_type='weight') # or 'gain', 'cover'
            # Normalize if desired
            # total = sum(importance.values())
            # importance = {k: v / total for k, v in importance.items()}
            logger.info(f"Calculated XGBoost feature importance: {importance}")
            return importance
        except Exception as e:
            logger.error(f"Error getting XGBoost feature importance: {e}")
            return None
    elif model_type in ['SVM', 'LSTM', 'SARIMA']:
        # Placeholder for SHAP or other methods
        logger.info(f"SHAP or other importance calculation needed for {model_type}")
        # Example SHAP (requires shap library and careful implementation):
        # try:
        #     import shap
        #     explainer = shap.KernelExplainer(model.predict, X) # Adjust based on model type
        #     shap_values = explainer.shap_values(X)
        #     # Process shap_values to get mean absolute importance per feature
        #     importance = pd.DataFrame(shap_values, columns=X.columns).abs().mean().to_dict()
        #     logger.info(f"Calculated SHAP importance for {model_type}: {importance}")
        #     return importance
        # except ImportError:
        #     logger.error("SHAP library not installed. Cannot calculate SHAP importance.")
        #     return None
        # except Exception as e:
        #     logger.error(f"Error calculating SHAP importance for {model_type}: {e}")
        #     return None
        return {"placeholder": 1.0} # Return dummy data for now
    else:
        return None
