import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV
import pymongo
from pymongo import MongoClient
import pickle
import os
import sys
import numpy as np
import argparse
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_regression
import shap  # Import SHAP library for model explainability

# Add parent directory to path to import mongo_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stocks.mongo_utils import get_mongo_db, STOCK_PRICES_COLLECTION, STOCKS_COLLECTION


# --- 1. Load Stock Price Data from MongoDB ---
def load_data_from_mongodb(mongo_uri, db_name, stock_symbol,
                           start_date=None, end_date=None):
    """Loads stock data from MongoDB, with optional date range."""
    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db['stock_prices']  # Use 'stock_prices' collection directly

        # Find the single document for the stock symbol
        stock_doc = collection.find_one({"symbol": stock_symbol})

        if not stock_doc:
            raise ValueError(
                f"No document found for symbol '{stock_symbol}' in MongoDB "
                f"collection 'stock_prices'."
            )

        historical_data = stock_doc.get("historical_data", [])
        if not historical_data:
             raise ValueError(
                f"No 'historical_data' found for symbol '{stock_symbol}'."
            )

        # Convert to DataFrame
        df = pd.DataFrame(historical_data)

        if 'Date' not in df.columns:
             # Attempt to find date column case-insensitively
             date_col = next((col for col in df.columns if col.lower() == 'date'), None)
             if not date_col:
                 raise ValueError(
                     f"Required field 'Date' (or similar) not found in 'historical_data'."
                 )
             df.rename(columns={date_col: 'Date'}, inplace=True)
            
        # Ensure 'Date' is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter by date range if provided
        if start_date and end_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]

        if df.empty:
            raise ValueError(
                f"No historical data found for symbol '{stock_symbol}' "
                f"within the specified date range ({start_date} to {end_date})."
            )

        # Check/Rename required columns (case-insensitive check)
        required_cols = ['Close', 'Volume', 'Open', 'High', 'Low']
        column_map = {}
        missing_cols = []
        current_cols_lower = {col.lower(): col for col in df.columns}

        for req_col in required_cols:
            req_col_lower = req_col.lower()
            if req_col_lower in current_cols_lower:
                original_case_col = current_cols_lower[req_col_lower]
                if original_case_col != req_col: # Only map if case is different
                     column_map[original_case_col] = req_col
            elif req_col not in df.columns: # Check if already correct case or truly missing
                 missing_cols.append(req_col)

        if column_map:
            df.rename(columns=column_map, inplace=True)

        if missing_cols:
             # Recheck after potential rename
             still_missing = [col for col in missing_cols if col not in df.columns]
             if still_missing:
                raise ValueError(
                    f"Missing required columns in historical_data: {', '.join(still_missing)}"
                )

        # Set Date as index
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True) # Ensure data is sorted by date

        # Convert numeric columns, coercing errors
        for col in ['Close', 'Volume', 'Open', 'High', 'Low']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['Close', 'Volume', 'Open', 'High', 'Low'], inplace=True) # Drop rows where essential numeric data is missing

        return df

    except pymongo.errors.ConnectionFailure as e:
        raise Exception(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        raise Exception(f"Error loading data from MongoDB: {e}")
    finally:
        if client:
            client.close()


# --- 2. Prepare Data for SVM ---
def prepare_data_for_svm(df, look_back=30, num_features_to_select=20):
    """Prepares data: creates features, scales, selects, and splits."""
    print(f"Preparing data with look_back={look_back}, num_features={num_features_to_select}")

    if 'Close' not in df.columns:
        raise ValueError("Required field 'Close' not found in DataFrame.")

    # 1. Feature Engineering
    df = df.copy()
    # Consider removing less critical features if still slow
    df['SMA'] = df['Close'].rolling(window=look_back).mean()
    df['EMA'] = df['Close'].ewm(span=look_back, adjust=False).mean()
    df['Price_Range'] = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].diff()
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    # 2. Create Lagged Features
    print("Creating lagged features...")
    for i in range(1, look_back + 1):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    print("Lagged features created.")

    # 3. Define Target Variable
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    # 4. Separate Features (X) and Target (y)
    X = df.drop('Target', axis=1)
    y = df['Target']

    # Check if X or y are empty after processing
    if X.empty or y.empty:
         raise ValueError("No data remaining after feature engineering and lag creation. Check look_back period and data length.")

    # 5. Feature Scaling
    print("Scaling features...")
    close_features = ['Close'] + [f'Close_Lag_{i}' for i in range(1, look_back + 1) if f'Close_Lag_{i}' in X.columns]
    other_features = [col for col in X.columns if col not in close_features]

    close_scaler = StandardScaler()
    # Use .loc to avoid SettingWithCopyWarning, ensure features exist
    X.loc[:, close_features] = close_scaler.fit_transform(X[close_features])

    other_scaler = None
    if other_features:
        other_scaler = StandardScaler()
        X.loc[:, other_features] = other_scaler.fit_transform(X[other_features])
    print("Feature scaling complete.")

    # 6. Feature Selection
    selected_features = X.columns.tolist() # Default if no selection needed
    if 0 < num_features_to_select < X.shape[1]:
        print(f"Selecting top {num_features_to_select} features...")
        # Ensure k is not greater than the number of features
        k_best = min(num_features_to_select, X.shape[1])
        if k_best > 0 :
             selector = SelectKBest(score_func=f_regression, k=k_best)
             try:
                 # Ensure y has the same index as X before fitting selector
                 y_aligned = y.loc[X.index]
                 X_selected = selector.fit_transform(X, y_aligned)
                 selected_features = X.columns[selector.get_support()].tolist()
                 X = pd.DataFrame(X_selected, index=X.index, columns=selected_features)
                 print(f"Selected features: {selected_features}")
             except ValueError as e:
                 print(f"Warning: Feature selection failed ({e}). Using all features.")
                 selected_features = X.columns.tolist()
        else:
             print("Warning: num_features_to_select is zero or negative. Skipping feature selection.")
             selected_features = X.columns.tolist()

    else:
      print("Skipping feature selection (num_features >= total features or <= 0).")
      selected_features = X.columns.tolist()


    # 7. Train-Test Split
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False # Time series data should not be shuffled
    )
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    if X_train.empty or y_train.empty:
         raise ValueError("Training set is empty after split. Check data length and test_size.")
    if X_test.empty or y_test.empty:
          raise ValueError("Test set is empty after split. Check data length and test_size.")


    # 8. Scale Target
    target_scaler = StandardScaler()
    # Ensure y_train/y_test are numpy arrays for scaler
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    print("Target variable scaled.")

    return (X_train, X_test, y_train_scaled, y_test_scaled, close_scaler,
            other_scaler, target_scaler, selected_features)


# --- Parse Command Line Arguments ---
def parse_args():
    """Parse command line arguments for training parameters."""
    parser = argparse.ArgumentParser(description='Train SVM model for stock price prediction (Optimized for Speed)')

    # Stock selection
    stock_group = parser.add_mutually_exclusive_group(required=True) # Make stock selection mandatory
    stock_group.add_argument('--symbol', type=str, help='Single stock symbol to train on (e.g., AAPL)')
    stock_group.add_argument('--symbols', type=str, help='Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL)')

    # Date range
    parser.add_argument('--start-date', type=str, default="2022-01-01", # Shorter default range
                      help='Start date for training data (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime("%Y-%m-%d"), # Default to today
                      help='End date for training data (YYYY-MM-DD format)')

    # Model parameters
    parser.add_argument('--look-back', type=int, default=30, # Reduced default
                      help='Look-back period for time series features (default: 30)')
    parser.add_argument('--features', type=int, default=20, # Reduced default
                      help='Number of features to select (default: 20, 0 to skip)')
    parser.add_argument('--kernel', type=str, default='rbf',
                      choices=['linear', 'poly', 'rbf', 'sigmoid'],
                      help='SVM kernel type (default: rbf)')
    # Removed C and gamma ranges as we use fixed parameters now
    parser.add_argument('--skip-shap', action='store_true', # Add flag to skip SHAP
                        help='Skip the computationally expensive SHAP value calculation.')

    return parser.parse_args()

# --- Helper function to get stock symbols ---
def get_stock_symbols(args):
    """Get stock symbols from arguments."""
    if args.symbol:
        return [args.symbol.upper()]
    elif args.symbols:
        # Handle potential empty strings if comma is at the end
        return [symbol.strip().upper() for symbol in args.symbols.split(',') if symbol.strip()]
    else:
        # This case should not be reached if the group is required=True
         print("Error: No stock symbol(s) provided.")
         sys.exit(1)


# --- 3. Train SVM Model with Kernel-Specific "Best Guess" Parameters ---
def train_svm_model(X_train, y_train, kernel='rbf'):
    """Trains SVM model with kernel-specific 'best guess' hyperparameters."""

    # --- Define "Best Guess" Parameters based on Kernel ---
    if kernel == 'linear':
        C = 1.0  # Common default for linear SVM
        gamma_to_use = 'scale' # Value ignored by linear kernel, but needs a valid setting
        print(f"Using kernel='linear' with C={C} (gamma is ignored)")
    elif kernel == 'poly':
        C = 10.0 # Often needs higher C than linear
        gamma_to_use = 'scale' # Data-dependent gamma
        print(f"Using kernel='poly' with C={C}, gamma='{gamma_to_use}' (and default degree=3)")
    elif kernel == 'sigmoid':
        C = 10.0
        gamma_to_use = 'scale'
        print(f"Using kernel='sigmoid' with C={C}, gamma='{gamma_to_use}'")
    else: # Default case for 'rbf' or any other unspecified kernel
        kernel = 'rbf' # Ensure kernel is set to rbf if not one of the above
        C = 10.0 # Moderate regularization for RBF
        gamma_to_use = 'scale' # 'scale' is often a good starting point for RBF
        print(f"Using kernel='rbf' with C={C}, gamma='{gamma_to_use}'")

    # Epsilon remains standard
    epsilon = 0.1

    print("Starting SVM model training...")
    # Instantiate SVR with the selected parameters
    model = SVR(kernel=kernel, C=C, gamma=gamma_to_use, epsilon=epsilon)

    # Fit the model directly
    model.fit(X_train, y_train)

    print("SVM model training completed.")
    return model


# --- 4. Evaluate Model ---
def evaluate_model(model, X_test, y_test, close_scaler, other_scaler, target_scaler, stock_symbol, start_date, end_date, evaluation_results_collection, skip_shap=False):
    """Evaluates the model, inverse transforms, stores results, optionally skips SHAP."""
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    # Inverse transform predictions and actual values to original scale
    y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate regression metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    # Calculate MAPE carefully avoiding division by zero
    mask = y_test_orig != 0
    mape = np.mean(np.abs((y_test_orig[mask] - y_pred_orig[mask]) / y_test_orig[mask])) * 100 if np.any(mask) else 0

    print("\nRegression Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R-squared (R2): {r2:.4f}")

    # Calculate classification metrics (predicting direction of change)
    accuracy = np.nan
    conf_matrix = np.array([])
    class_report = "Not calculated (requires at least 2 price changes)."

    # Ensure there are enough points to calculate differences
    if len(y_test_orig) > 1 and len(y_pred_orig) > 1:
        y_test_diff = np.diff(y_test_orig)
        y_pred_diff = np.diff(y_pred_orig)

        # Handle cases where diff results in zero length (if input was len 1) - though check above should prevent this
        if len(y_test_diff) > 0 and len(y_pred_diff) > 0:
            y_test_class = np.where(y_test_diff > 0, 1, 0)
            y_pred_class = np.where(y_pred_diff > 0, 1, 0)

            # Ensure classes have the same length (should be len(y_test_orig) - 1)
            min_len = min(len(y_test_class), len(y_pred_class))
            y_test_class = y_test_class[:min_len]
            y_pred_class = y_pred_class[:min_len]

            if len(y_test_class) > 0: # Check if there are samples to evaluate
                accuracy = accuracy_score(y_test_class, y_pred_class)
                conf_matrix = confusion_matrix(y_test_class, y_pred_class)
                # Ensure there are predicted samples before generating report
                if np.sum(conf_matrix) > 0:
                     # Define labels, handle case with only one class prediction
                     unique_labels = np.unique(np.concatenate((y_test_class, y_pred_class)))
                     target_names = ['Down/Same', 'Up']
                     report_labels = [0, 1] if len(unique_labels) == 2 else unique_labels
                     report_target_names = [target_names[i] for i in report_labels]

                     class_report = classification_report(y_test_class, y_pred_class,
                                                         labels=report_labels,
                                                         target_names=report_target_names,
                                                         zero_division=0)
                else:
                     class_report = "No predictions made for classification."


                print("\nClassification Metrics (Price Change Direction):")
                print(f"  Accuracy: {accuracy:.4f}")
                print("  Confusion Matrix (Rows=Actual, Cols=Predicted):\n", conf_matrix)
                print("  Classification Report:\n", class_report)
            else:
                print("\nClassification Metrics: Not enough data points with changes to evaluate direction accuracy.")
                accuracy = np.nan
                conf_matrix = np.array([])
                class_report = "Not enough data points with changes."
        else:
             print("\nClassification Metrics: Not enough data points to calculate price differences.")

    else:
        print("\nClassification Metrics: Not enough data points in test set for classification.")


    # Generate SHAP values (Optional and Reduced Sample Size)
    feature_importance_dict = {}
    if not skip_shap:
        print("\nGenerating SHAP values for model explanation (using small sample)...")
        try:
            # DRASTICALLY reduce samples for speed. KernelExplainer is SLOW.
            max_samples_shap = min(10, X_test.shape[0]) # Use only 10 samples!
            if max_samples_shap > 0:

                 # Create a SHAP KernelExplainer for the SVM model
                 # Use shap.sample to get a representative sample
                 X_test_sample = shap.sample(X_test, max_samples_shap)

                 # Define a prediction function wrapper if necessary (sometimes needed for sklearn)
                 def predict_fn(data):
                     # Ensure data is DataFrame with correct columns if needed by model
                     if not isinstance(data, pd.DataFrame):
                          data_df = pd.DataFrame(data, columns=X_test.columns)
                     else:
                          data_df = data
                     return model.predict(data_df)

                 # Use the prediction function with the explainer
                 explainer = shap.KernelExplainer(predict_fn, X_test_sample)

                 # Calculate SHAP values (this is the slow part)
                 # Use nsamples='auto' or a small number for faster estimation
                 shap_values = explainer.shap_values(X_test_sample, nsamples=50) # Use fewer nsamples

                 # Check if shap_values is a list (multi-output) or single array
                 if isinstance(shap_values, list):
                      # Handle multi-output case if necessary, e.g., take the first output
                      shap_importance = np.abs(shap_values[0]).mean(axis=0)
                      print("Warning: SHAP values appear multi-output, using first output for importance.")
                 else:
                      shap_importance = np.abs(shap_values).mean(axis=0)


                 for i, importance in enumerate(shap_importance):
                    # Ensure feature index is valid
                     if i < len(X_test.columns):
                         feature_importance_dict[X_test.columns[i]] = float(importance)
                     else:
                         print(f"Warning: SHAP value index {i} out of bounds for features.")


                 # Print top features by SHAP importance
                 print("\nSHAP Feature Importance (Top Features on Sample):")
                 sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                 for feature, importance in sorted_features:
                     print(f"  {feature}: {importance:.4f}")
            else:
                 print("Skipping SHAP: Not enough samples in the test set.")


        except Exception as e:
            import traceback
            print(f"Error generating SHAP values: {e}")
            # print(traceback.format_exc()) # Uncomment for detailed debugging
            feature_importance_dict = {}
    else:
        print("\nSkipping SHAP value calculation as requested.")


    # Store evaluation results
    actual_params = model.get_params() # Get the actual parameters used
    evaluation_data = {
        "stock_symbol": stock_symbol,
        "train_start_date": start_date.isoformat(),
        "train_end_date": end_date.isoformat(),
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_type": "SVM",
        "svr_kernel": actual_params.get('kernel', 'unknown'), # Use actual params
        "svr_c": actual_params.get('C', np.nan),
        "svr_epsilon": actual_params.get('epsilon', np.nan),
        "svr_gamma": actual_params.get('gamma', 'unknown'), # Gamma might be string ('scale') or float
        "look_back_period": getattr(model, 'look_back', None), # Get look_back if saved on model
        "num_features_selected": len(X_test.columns) if X_test is not None else None,
        "selected_features_list": X_test.columns.tolist() if X_test is not None else None,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape if mape is not None else np.nan,
        "r2": r2,
        "direction_accuracy": accuracy if not np.isnan(accuracy) else None, # Store None if NaN
        "confusion_matrix": conf_matrix.tolist() if conf_matrix.size > 0 else [],
        "classification_report": class_report,
        "shap_feature_importance": feature_importance_dict,
        "shap_skipped": skip_shap,
    }
    try:
        evaluation_results_collection.insert_one(evaluation_data)
        print(f"Evaluation results for {stock_symbol} stored in MongoDB.")
    except Exception as e:
        print(f"Error storing evaluation results in MongoDB: {e}")

    # Return key metrics
    return mse, rmse, mae, r2, mape, accuracy


# --- 5. Save Trained Model ---
def save_model(model, stock_symbol, close_scaler, other_scaler, target_scaler, selected_features, model_name="svm-fast-bestguess"): # Updated default name
    """Saves the model and scalers."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Ensure base directory exists relative to the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_SAVE_DIR_BASE = os.path.join(script_dir, '..', 'train-model') # Go up one level from script location
    model_dir = os.path.join(MODEL_SAVE_DIR_BASE, stock_symbol, f"{model_name}-{timestamp}")

    try:
         os.makedirs(model_dir, exist_ok=True)
         print(f"Created model directory: {model_dir}")
    except OSError as e:
         print(f"Error creating directory {model_dir}: {e}")
         # Fallback: save in the script's directory
         model_dir = os.path.join(script_dir, f"{stock_symbol}-{model_name}-{timestamp}")
         print(f"Attempting to save in fallback directory: {model_dir}")
         os.makedirs(model_dir, exist_ok=True) # Create fallback if it doesn't exist

    model_path = os.path.join(model_dir, "model.pkl")
    close_scaler_path = os.path.join(model_dir, "close_scaler.pkl")
    other_scaler_path = os.path.join(model_dir, "other_scaler.pkl")
    target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
    features_path = os.path.join(model_dir, "selected_features.pkl") # Save selected features list

    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Trained SVM model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")

    # Save scalers and features
    if close_scaler:
         try:
             with open(close_scaler_path, 'wb') as file:
                 pickle.dump(close_scaler, file)
             print(f"Close scaler saved to: {close_scaler_path}")
         except Exception as e:
             print(f"Error saving close scaler to {close_scaler_path}: {e}")

    if other_scaler:
        try:
            with open(other_scaler_path, 'wb') as file:
                pickle.dump(other_scaler, file)
            print(f"Other features scaler saved to: {other_scaler_path}")
        except Exception as e:
             print(f"Error saving other scaler to {other_scaler_path}: {e}")

    if target_scaler:
        try:
            with open(target_scaler_path, 'wb') as file:
                pickle.dump(target_scaler, file)
            print(f"Target scaler saved to: {target_scaler_path}")
        except Exception as e:
            print(f"Error saving target scaler to {target_scaler_path}: {e}")

    if selected_features:
         try:
            with open(features_path, 'wb') as file:
                 pickle.dump(selected_features, file)
            print(f"Selected features saved to: {features_path}")
         except Exception as e:
             print(f"Error saving selected features list to {features_path}: {e}")


# --- Main Program ---
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # --- MongoDB Connection Details ---
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGO_DB_NAME', 'stock_prices')
    # Changed collection name slightly to distinguish these results
    evaluation_collection_name = "svm_evaluation_results_bestguess"

    # --- Date Range ---
    start_date_str = args.start_date
    end_date_str = args.end_date
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        if start_date >= end_date:
             print("Error: Start date must be before end date.")
             sys.exit(1)
    except ValueError as e:
        print(f"Error parsing dates: {e}. Please use YYYY-MM-DD format.")
        sys.exit(1)

    print(f"Training FAST SVM (BestGuess Params) with date range: {start_date} to {end_date}")

    # --- Load Stock Symbols ---
    stock_symbols = get_stock_symbols(args)
    print(f"Processing {len(stock_symbols)} stock(s): {', '.join(stock_symbols)}")

    # --- Model Parameters from Args ---
    look_back = args.look_back
    num_features = args.features
    kernel_choice = args.kernel # Get user's kernel choice
    skip_shap = args.skip_shap

    print(f"Using look_back={look_back}, num_features={num_features}, kernel={kernel_choice}")
    if skip_shap:
        print("SHAP calculation will be skipped.")

    client = None
    try:
        # Establish MongoDB connection once
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000) # Add timeout
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("MongoDB connection successful.")
        db = client[db_name]
        evaluation_results_collection = db[evaluation_collection_name]
        print(f"Storing results in '{db_name}.{evaluation_collection_name}'")


        for stock_symbol_to_predict in stock_symbols:
            print(f"\n--- Processing Stock: {stock_symbol_to_predict} ---")
            try:
                # 1. Load Data
                print("Loading data...")
                df = load_data_from_mongodb(
                    mongo_uri, db_name,
                    stock_symbol_to_predict, start_date, end_date
                )
                print(f"Loaded {df.shape[0]} data points.")

                # Check for sufficient data *after* loading
                min_required_data = look_back + 5 # Heuristic: lookback + buffer for train/test
                if df.shape[0] < min_required_data:
                     print(f"Skipping {stock_symbol_to_predict}: Insufficient data ({df.shape[0]} points) for look_back={look_back}. Need at least {min_required_data}.")
                     continue

                # 2. Prepare Data
                print("Preparing data...")
                (X_train, X_test, y_train_scaled, y_test_scaled,
                 close_scaler, other_scaler, target_scaler, selected_features) = prepare_data_for_svm(
                    df, look_back=look_back,
                    num_features_to_select=num_features
                )

                # 3. Train Model (Pass the chosen kernel)
                svm_model = train_svm_model(X_train, y_train_scaled, kernel=kernel_choice) # Pass kernel here

                # Optionally store look_back on the model instance for saving later
                svm_model.look_back = look_back


                # 4. Evaluate and Store Results
                evaluate_model(svm_model, X_test, y_test_scaled, close_scaler,
                               other_scaler, target_scaler,
                               stock_symbol_to_predict, start_date, end_date,
                               evaluation_results_collection, skip_shap=skip_shap)

                # 5. Save Model and Scalers
                print("Saving model and scalers...")
                save_model(svm_model, stock_symbol_to_predict, close_scaler,
                           other_scaler, target_scaler, selected_features)

            except ValueError as ve:
                print(f"Value Error processing {stock_symbol_to_predict}: {ve}. Skipping stock.")
                continue
            except pymongo.errors.OperationFailure as op_error:
                 print(f"MongoDB Operation Failure for {stock_symbol_to_predict}: {op_error}. Check permissions or query. Skipping stock.")
                 continue
            except Exception as e:
                import traceback
                print(f"An unexpected error occurred for {stock_symbol_to_predict}: {e}")
                print(traceback.format_exc()) # Print detailed traceback
                print("Skipping stock.")
                continue

    except pymongo.errors.ConnectionFailure as e:
        print(f"Fatal Error: Could not connect to MongoDB at {mongo_uri}. Details: {e}")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred in the main program: {e}")
        print(traceback.format_exc()) # Print detailed traceback
    finally:
        if client:
            client.close()
            print("MongoDB connection closed.")

    print("\n--- FAST SVM (BestGuess Params) Stock Price Prediction and Evaluation Completed. ---")