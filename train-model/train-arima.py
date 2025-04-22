# train-arima.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
import pymongo
from pymongo import MongoClient
import pickle
import os
import sys
import argparse
from tqdm import tqdm  # Import tqdm for progress bar
from datetime import datetime, timezone

# Add parent directory to path to import mongo_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stocks.mongo_utils import get_mongo_db, STOCK_PRICES_COLLECTION, STOCKS_COLLECTION

# --- Parse Command Line Arguments ---
def parse_args():
    """Parse command line arguments for training parameters."""
    parser = argparse.ArgumentParser(description='Train ARIMA model for stock price prediction')
    
    # Stock selection
    stock_group = parser.add_mutually_exclusive_group()
    stock_group.add_argument('--symbol', type=str, help='Single stock symbol to train on (e.g., AAPL)')
    stock_group.add_argument('--symbols', type=str, help='Comma-separated list of stock symbols to train on (e.g., AAPL,MSFT,GOOGL)')
    
    # Date range
    parser.add_argument('--start-date', type=str, default="2020-02-25", 
                      help='Start date for training data (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str, default="2025-02-25",
                      help='End date for training data (YYYY-MM-DD format)')
    
    # Model parameters
    parser.add_argument('--price-field', type=str, default="close",
                      help='Price field to predict (default: close)')
    parser.add_argument('--max-p', type=int, default=3,
                      help='Maximum p value for ARIMA (default: 3)')
    parser.add_argument('--max-q', type=int, default=3,
                      help='Maximum q value for ARIMA (default: 3)')
    parser.add_argument('--auto-diff', action='store_true',
                      help='Automatically determine differencing order')
    parser.add_argument('--diff-order', type=int, default=1,
                      help='Differencing order if auto-diff is False (default: 1)')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Test size as fraction of data (default: 0.2)')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                      help='Skip feature engineering step')
                      
    return parser.parse_args()

# --- Helper function to get stock symbols ---
def get_stock_symbols(args):
    """Get stock symbols from arguments or CSV file."""
    if args.symbol:
        return [args.symbol.upper()]
    elif args.symbols:
        return [symbol.strip().upper() for symbol in args.symbols.split(',')]
    else:
        # Default: load from CSV
        stocks_file = "stock_symbols.csv"
        try:
            stocks_df = pd.read_csv(stocks_file)
            return stocks_df['Symbol'].tolist()
        except FileNotFoundError:
            print(f"Error: {stocks_file} not found.")
            sys.exit(1)
        except KeyError:
            print(f"Error: 'Symbol' column not found in {stocks_file}.")
            sys.exit(1)

# --- Feature Engineering Functions ---
def calculate_moving_average(series, window=10):
    """Calculates the simple moving average."""
    return series.rolling(window=window).mean()

def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index."""
    delta = series.diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=period).mean()
    roll_down1 = np.abs(down.ewm(span=period).mean())
    RS = roll_up1 / roll_down1
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """Calculates Moving Average Convergence Divergence."""
    ema_fast = series.ewm(span=fast_period).mean()
    ema_slow = series.ewm(span=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_atr(df, period=14):
    """Calculates Average True Range."""
    high_low = df['High'] - df['Low']
    high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
    low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# --- 1. Load Stock Price Data from MongoDB ---
def load_data_from_mongodb(mongo_uri, db_name, stock_symbol, price_field='close', start_date=None, end_date=None, feature_engineering=True):
    """Loads stock price data from MongoDB, with optional date range and feature engineering."""
    client = None  # Initialize client outside the try block
    try:
        # First try the direct approach that works with XGBoost
        client = MongoClient(mongo_uri)
        db = client[db_name]
        
        # Try both collection names and variations of symbol case
        collections_to_try = ['stock_prices', 'stockPrices']
        symbol_variations = [stock_symbol, stock_symbol.lower(), stock_symbol.upper()]
        
        found = False
        for collection_name in collections_to_try:
            collection = db[collection_name]
            for symbol_var in symbol_variations:
                stock_doc = collection.find_one({"symbol": symbol_var})
                if stock_doc:
                    found = True
                    break
            if found:
                break
                
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
             raise ValueError(
                f"Required field 'Date' not found in 'historical_data'."
            )
            
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

        # Map column names to expected format
        column_map = {
            'date': 'Date',
            'close': 'Close', 
            'volume': 'Volume',
            'open': 'Open',
            'high': 'High',
            'low': 'Low'
        }
        df.rename(columns=column_map, inplace=True, errors='ignore')
        
        # Also adjust price_field if it was renamed
        if price_field == 'close':
            price_field = 'Close'
            
        # Check for required columns after renaming
        required_cols = ['Date', 'Close', 'Volume', 'Open', 'High', 'Low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in historical_data: {', '.join(missing_cols)}"
            )

        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        if feature_engineering:
            df['MA_10'] = calculate_moving_average(df['Close'], window=10)
            df['RSI'] = calculate_rsi(df['Close'])
            macd_line, signal_line, macd_histogram = calculate_macd(df['Close'])
            df['MACD_Line'] = macd_line
            df['MACD_Signal'] = signal_line
            df['ATR'] = calculate_atr(df)
            df['Price_Range'] = df['High'] - df['Low']
            df['Volume_Change'] = df['Volume'].diff()

            df.dropna(inplace=True) # Important: Drop rows with NaN after feature engineering

        price_series = df[price_field].squeeze() # Still predicting price based on selected field

        return price_series, df # Return both price_series and the feature-rich DataFrame

    except pymongo.errors.ConnectionFailure as e:
        raise Exception(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        raise Exception(f"Error loading data from MongoDB: {e}")
    finally:
        if client: # Check if client is defined before trying to close
            client.close()

# --- 2. Check for Stationarity using ADF Test ---
def check_stationarity(series):
    """Performs Augmented Dickey-Fuller test to check for stationarity."""
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    if dfoutput['p-value'] <= 0.05:
        return True
    else:
        return False

# --- 3. Make Series Stationary (if needed) using Differencing ---
def make_stationary(series, d=1):
    """Makes a time series stationary by differencing."""
    if not check_stationarity(series):
        diff_series = series.diff(d).dropna()
        if check_stationarity(diff_series):
            return diff_series, d
        else:
            return None, d
    else:
        return series, 0

# --- 4. Split Data into Training and Testing Sets ---
def train_test_split(data, test_size=0.2):
    """Splits time series data into training and testing sets."""
    if not isinstance(data, pd.Series): # Modified to accept Series or DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas Series or DataFrame.")

    if isinstance(data, pd.Series):
        split_index = int(len(data) * (1 - test_size))
        train_data, test_data = data[:split_index], data[split_index:]
    elif isinstance(data, pd.DataFrame):
        split_index = int(len(data) * (1 - test_size))
        train_data, test_data = data[:split_index], data[split_index:]
    return train_data, test_data

# --- 5. Automatically Select ARIMA (p, d, q) parameters using AIC ---
def get_auto_arima_params(train_series, max_p=3, max_q=3):
    """Automatically defines ARIMA parameters (p, d, q) using AIC minimization."""
    best_aic = float("inf")
    best_order = None
    best_model_fit = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            order = (p, 0, q) # d is already determined by stationarity check
            try:
                model = ARIMA(train_series, order=order)
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_model_fit = model_fit
            except Exception as e: # Catch any potential errors during model fitting
                continue # Just continue to the next combination if there's an issue

    if best_order:
        print(f"Automated ARIMA parameter selection - Best Order: ARIMA{best_order} with AIC={best_aic:.2f}")
        return best_order
    else:
        print("Automated ARIMA parameter selection failed to find a suitable order. Using default (1,1,1).")
        return (1, 1, 1) # Return default order if auto selection fails

# --- 6. Train ARIMA Model ---
def train_arima_model(train_series, order):
    """Trains an ARIMA model on the training data."""
    print("Starting ARIMA model training...")  # Progress message
    try:
        model = ARIMA(train_series, order=order)
        model_fit = model.fit()
        print("ARIMA model training finished.") # Progress message
        return model_fit
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        return None

# --- 7. Evaluate Model on Test Set (Walk-Forward Validation) ---
def evaluate_model(model_fit, train_series, test_series, original_series, diff_order, stock_symbol, order, evaluation_results_collection, start_date, end_date):
    """Evaluates the trained ARIMA model using walk-forward validation and stores results in MongoDB."""
    history = list(train_series)
    predictions = []
    for t in tqdm(range(len(test_series)), desc="Walk-Forward Validation"): # Added tqdm progress bar
        model = ARIMA(history, order=order)
        model_fit_wf = model.fit()
        output = model_fit_wf.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test_series[t])

    if diff_order > 0:
        predictions_original_scale = []
        history_original_scale = list(original_series[:len(train_series)])

        for i in range(len(predictions)):
            yhat_original_scale = predictions[i] + history_original_scale[-1]
            predictions_original_scale.append(yhat_original_scale)
            history_original_scale.append(original_series[len(train_series) + i])
    else:
        predictions_original_scale = predictions
        history_original_scale = list(original_series)

    # Correct the start index for actual values in original scale
    start_index = len(train_series) + diff_order
    actuals_original_scale = original_series[start_index:].values

    # Ensure the lengths match the test series length
    actuals_original_scale = actuals_original_scale[:len(test_series)]
    predictions_original_scale = predictions_original_scale[:len(test_series)]

    # Calculate regression metrics
    rmse_val = np.sqrt(mean_squared_error(actuals_original_scale, predictions_original_scale))
    mae_val = mean_absolute_error(actuals_original_scale, predictions_original_scale)
    mape_val = np.mean(np.abs((actuals_original_scale - predictions_original_scale) / actuals_original_scale)) * 100

    print(f'\nModel Evaluation (Original Scale - Regression Metrics):')
    print(f'RMSE: {rmse_val:.2f}')
    print(f'MAE: {mae_val:.2f}')
    print(f'MAPE: {mape_val:.2f}%')

    # Calculate classification metrics
    actual_directions = np.diff(actuals_original_scale) > 0
    predicted_directions = np.diff(predictions_original_scale) > 0

    # Calculate confusion matrix and other metrics
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    cm = confusion_matrix(actual_directions, predicted_directions)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(actual_directions, predicted_directions)
    recall = recall_score(actual_directions, predicted_directions)
    f1 = f1_score(actual_directions, predicted_directions)

    print(f'\nModel Evaluation (Direction Prediction - Classification Metrics):')
    print("Confusion Matrix:")
    print(cm)
    print(f'Accuracy: {accuracy_score(actual_directions, predicted_directions):.2f}')
    print(f'Sensitivity (Recall or True Positive Rate): {sensitivity:.2f}')
    print(f'Specificity (True Negative Rate): {specificity:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1-Score: {f1:.2f}')

    # Generate SHAP values for time series interpretation
    print("\nGenerating SHAP interpretations for ARIMA model...")
    feature_importance_dict = {}
    try:
        # For ARIMA models, we use a different approach than ML models
        # We'll examine how different components (AR, I, MA) contribute to the forecast
        
        # 1. Get model coefficients
        model_params = model_fit.params()
        
        # 2. Interpret AR coefficients - past observations
        ar_coefs = model_params.get('ar', [])
        if isinstance(ar_coefs, np.ndarray):
            ar_coefs = ar_coefs.tolist()
        elif not isinstance(ar_coefs, list):
            ar_coefs = []
        
        # 3. Interpret MA coefficients - past errors
        ma_coefs = model_params.get('ma', [])
        if isinstance(ma_coefs, np.ndarray):
            ma_coefs = ma_coefs.tolist()
        elif not isinstance(ma_coefs, list):
            ma_coefs = []
        
        # 4. Calculate component contributions for the last few predictions
        sample_size = min(20, len(predictions))
        last_predictions = predictions[-sample_size:]
        
        # Approximate component importance by analyzing residuals
        if hasattr(model_fit, 'resid'):
            residuals = model_fit.resid
            
            # Calculate lag correlations to determine feature importance
            lag_corr = []
            for lag in range(1, min(10, len(residuals))):
                corr = np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1]
                lag_corr.append((f"Lag_{lag}", abs(corr)))
            
            # Sort by correlation strength
            lag_corr.sort(key=lambda x: x[1], reverse=True)
            
            # Create feature importance dictionary
            for lag, importance in lag_corr:
                feature_importance_dict[lag] = float(importance)
                
            # Print top lag influences
            print("\nARIMA Component Importance (Top Lags):")
            for lag, importance in lag_corr[:5]:  # Show top 5
                print(f"  {lag}: {importance:.4f}")
                
            # Also add order components to the feature importance
            feature_importance_dict["AR_component"] = float(order[0])
            feature_importance_dict["I_component"] = float(order[1])
            feature_importance_dict["MA_component"] = float(order[2])
            
            # Add coefficient information
            for i, coef in enumerate(ar_coefs):
                feature_importance_dict[f"AR_coef_{i+1}"] = float(coef)
            for i, coef in enumerate(ma_coefs):
                feature_importance_dict[f"MA_coef_{i+1}"] = float(coef)
                
    except Exception as e:
        print(f"Error generating SHAP values for ARIMA: {e}")
        # Provide at least the model order as feature importance
        feature_importance_dict = {
            "AR_component": float(order[0]),
            "I_component": float(order[1]),
            "MA_component": float(order[2])
        }

    # --- Store Trained Model ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = f"/home/skylap/Downloads/stockmarketprediction/train-model/{stock_symbol}/arima-{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")

    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model_fit, file)
        print(f"\nTrained ARIMA model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")

    # --- Store Evaluation Results in MongoDB ---
    evaluation_data = {
        "stock_symbol": stock_symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "arima_order": str(order),
        "rmse": rmse_val,
        "mae": mae_val,
        "mape": mape_val,
        "accuracy": accuracy_score(actual_directions, predicted_directions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(), # Store confusion matrix as list for MongoDB
        "component_importance": feature_importance_dict  # Store ARIMA component importance
    }
    try:
        evaluation_results_collection.insert_one(evaluation_data)
        print(f"Evaluation results for {stock_symbol} stored in MongoDB.")
    except Exception as e:
        print(f"Error storing evaluation results in MongoDB: {e}")


# --- Main Program ---
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # --- MongoDB Connection Details ---
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "stock_market_db"
    evaluation_collection_name = "arima_evaluation_results"

    # --- Model Parameters ---
    price_field_to_use = args.price_field
    feature_engineering = not args.skip_feature_engineering
    max_p = args.max_p
    max_q = args.max_q
    test_size = args.test_size
    
    print(f"ARIMA parameters: price_field={price_field_to_use}, feature_engineering={feature_engineering}")
    print(f"max_p={max_p}, max_q={max_q}, test_size={test_size}")

    # --- Date Range for Training and Prediction ---
    start_date_str = args.start_date
    end_date_str = args.end_date
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Please use YYYY-MM-DD format for dates.")
        sys.exit(1)
        
    print(f"Training with date range: {start_date} to {end_date}")

    # --- Load Stock Symbols ---
    stock_symbols = get_stock_symbols(args)
    print(f"Training ARIMA model for {len(stock_symbols)} stock(s): {', '.join(stock_symbols) if len(stock_symbols) < 5 else ', '.join(stock_symbols[:5]) + '...'}")

    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        evaluation_results_collection = db[evaluation_collection_name]

        for stock_symbol_to_predict in stock_symbols:
            print(f"\n--- Processing Stock: {stock_symbol_to_predict} ---")

            try:
                # 1. Load Data from MongoDB with Date Range and Feature Engineering
                stock_series, feature_df = load_data_from_mongodb(
                    mongo_uri, db_name, stock_symbol_to_predict, 
                    price_field_to_use, start_date, end_date, 
                    feature_engineering=feature_engineering
                )

                # 2. Handle Stationarity (using price series)
                if args.auto_diff:
                    stationary_series, diff_order = make_stationary(stock_series.copy())
                    if stationary_series is None:
                        print("Failed to make series stationary. Skipping stock.")
                        continue
                else:
                    # Use fixed differencing order
                    diff_order = args.diff_order
                    stationary_series = stock_series.diff(diff_order).dropna()

                # 3. Split Data
                train_data, test_data = train_test_split(stationary_series, test_size=test_size)
                original_train, original_test = train_test_split(stock_series, test_size=test_size)
                
                if feature_engineering:
                    feature_train_df, feature_test_df = train_test_split(feature_df, test_size=test_size)

                # 4. Automatically Set ARIMA Parameters
                best_order = get_auto_arima_params(train_data, max_p=max_p, max_q=max_q)

                # 5. Train ARIMA Model
                arima_model_fit = train_arima_model(train_data, best_order)
                if arima_model_fit is None:
                    print("ARIMA model training failed. Skipping stock.")
                    continue # Skip to the next stock symbol

                # 6. Evaluate Model (includes classification metrics and model saving & result storing)
                evaluate_model(arima_model_fit, train_data, test_data, stock_series, diff_order, stock_symbol_to_predict, best_order, evaluation_results_collection, start_date, end_date)

            except ValueError as ve:
                print(f"Value Error for {stock_symbol_to_predict}: {ve}. Skipping stock.")
                continue # Skip to the next stock symbol
            except Exception as e:
                print(f"An error occurred for {stock_symbol_to_predict}: {e}. Skipping stock.")
                continue # Skip to the next stock symbol


    except pymongo.errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main program: {e}")
    finally:
        if client:
            client.close()
    print("\n--- Stock Price Prediction and Evaluation Completed for all symbols. ---")