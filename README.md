# StockWise:Stock Market Prediction & Analysis

StockWise is a comprehensive web application for stock market analysis, prediction, and tracking. It provides users with real-time stock data, news sentiment analysis, and portfolio management capabilities.

![StockWise Logo](https://raw.githubusercontent.com/username/stockmarketprediction/main/static/img/logo.png)

## 📊 Features

- **Real-time Stock Data**: Access current stock prices, historical data, and key market indicators
- **News Sentiment Analysis**: Analyze news articles to gauge market sentiment for individual stocks
- **User Watchlists**: Create and manage custom watchlists of stocks you're interested in
- **Portfolio Tracking**: Track your investments and monitor performance
- **Stock Discovery**: Search and find new investment opportunities
- **Price Predictions**: Advanced algorithms to forecast potential price movements
- **Interactive Charts**: Visualize stock performance over custom time periods
- **Admin Dashboard**: Comprehensive admin interface for data management and updates

## 🧰 Tech Stack

- **Backend**: Django 3.1+
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Databases**:
  - SQLite (User authentication & app data)
  - MongoDB (Stock historical data & news)
- **APIs**:
  - yfinance (Yahoo Finance)
  - Finnhub (Financial News)
- **Data Analysis**:
  - pandas, numpy
  - scikit-learn, TensorFlow
- **Deployment**: Heroku-ready configuration

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MongoDB (local or remote)
- pip

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/username/stockmarketprediction.git
cd stockmarketprediction
```

2. **Set up a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root with the following:

```
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=stock_data

# Finnhub API Keys (for news data)
FINNHUB_API_KEYS=your_api_key_here
```

5. **Run migrations**

```bash
python manage.py migrate
```

6. **Load initial stock data**

```bash
python manage.py import_stocks
```

7. **Create a superuser**

```bash
python manage.py createsuperuser
```

8. **Run the development server**

```bash
python manage.py runserver
```

The application will be available at http://127.0.0.1:8000/

## 🔧 Detailed Setup Guide

### System Requirements
- **OS**: Linux, macOS, or Windows
- **RAM**: 4GB minimum (8GB+ recommended for model training)
- **Storage**: 1GB for application, plus additional space for stock data (5GB+ recommended)
- **Processor**: Multi-core processor recommended for model training

### Complete Installation Steps

#### 1. Database Setup

**SQLite Database** (Already configured for development):
- The SQLite database (db.sqlite3) is automatically created during migration
- No additional setup required for development environment

**MongoDB Setup**:
```bash
# Install MongoDB if not already installed
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS (using Homebrew)
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB service
# Ubuntu/Debian
sudo systemctl start mongodb

# macOS
brew services start mongodb-community

# Verify MongoDB is running
mongo --eval "db.version()"
```

#### 2. API Keys and External Services

1. **Finnhub API** (for news data):
   - Register at https://finnhub.io/
   - Get your API key from the dashboard
   - Add to your `.env` file as `FINNHUB_API_KEYS`

2. **Yahoo Finance** (no API key needed):
   - yfinance library is used and doesn't require authentication
   - However, be mindful of rate limits for production use

3. **Optional APIs** (for additional features):
   - Alpha Vantage for additional financial data (https://www.alphavantage.co/)
   - NewsAPI for broader news sources (https://newsapi.org/)

#### 3. Model Training

The application uses several prediction models. To train them:

```bash
# Activate your virtual environment first
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Update stock data before training
python manage.py update_stocks

# Train individual models
cd train-model
python train.py --model xgboost --symbol AAPL
python train.py --model lstm --symbol MSFT
python train.py --model arima --symbol GOOG

# Or train all models in batch mode
python train.py --batch training_batch.json
```

Model files will be saved in the `/train-model/{SYMBOL}/{model-type}/` directories.

#### 4. Running in Production

For production deployment:

1. **Set DEBUG=False in your environment variables**
   ```
   DEBUG=False
   ```

2. **Configure a proper database** (PostgreSQL recommended):
   ```
   DATABASE_URL=postgresql://user:password@localhost/stockwise
   ```

3. **Set up static files serving**:
   ```bash
   python manage.py collectstatic
   ```

4. **Use a production-ready web server**:
   ```bash
   # Install Gunicorn
   pip install gunicorn

   # Run with Gunicorn
   gunicorn stockmarketprediction.wsgi:application
   ```

5. **Set up a reverse proxy** with Nginx or Apache

#### 5. Scheduled Tasks

Set up cron jobs or scheduled tasks for data updates:

```bash
# Example crontab entries (Linux/macOS)

# Update stock prices daily at 18:00 (after market close)
0 18 * * 1-5 cd /path/to/stockmarketprediction && /path/to/venv/bin/python manage.py update_stocks

# Update news and sentiment analysis every 3 hours during market hours
0 9,12,15 * * 1-5 cd /path/to/stockmarketprediction && /path/to/venv/bin/python manage.py update_news

# Update commodities data daily
0 19 * * * cd /path/to/stockmarketprediction && /path/to/venv/bin/python manage.py update_commodities
```

### Troubleshooting

**MongoDB Connection Issues**:
- Verify MongoDB is running: `sudo systemctl status mongodb`
- Check your MONGO_URI in .env file
- Ensure your firewall allows connections to MongoDB port (default: 27017)

**Missing Stock Data**:
- Run `python manage.py import_stocks` to populate initial data
- Check `stock_symbols.csv` contains the symbols you want to track

**Model Training Errors**:
- Ensure you have at least 2 years of historical data for each stock
- Check `train_jobs.log` for specific training errors
- Increase memory allocation if you encounter memory errors

**API Rate Limits**:
- Implement retries with exponential backoff in production
- Consider upgrading to paid API tiers for higher limits
- Spread requests across multiple API keys when possible

### Updating the Application

```bash
# Pull the latest changes
git pull origin main

# Install any new dependencies
pip install -r requirements.txt

# Apply any new migrations
python manage.py migrate

# Restart the application
# (restart method depends on your deployment setup)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📝 Acknowledgements

- Yahoo Finance for stock data API
- Finnhub for financial news
- All open source libraries used in this project

## 📊 Data Updates

The application includes several management commands to update data:

- Update all stock prices: `python manage.py update_stocks`
- Update commodities prices: `python manage.py update_commodities`
- Fetch latest news: `python manage.py update_news`

You can also schedule these commands using a cron job or Heroku scheduler.

## 📱 User Features

### Registration and Login
- Create a new account to access personalized features
- Secure authentication system
- Profile management

### Stock Search
- Search for stocks by symbol or company name
- View detailed information, charts and news
- Add stocks to your watchlist

### Watchlist Management
- Create and customize your watchlist
- Track stock performance
- Add notes and purchase information for your investments

### News & Sentiment
- View latest news for each stock
- Sentiment analysis categorizes news as positive, negative, or neutral
- Track news impact on stock price

## 👨‍💻 Admin Features

The admin dashboard provides additional features:

- Fetch news for specific stocks
- Update price data
- Manage users and watchlists
- View application statistics

Access the admin interface at http://127.0.0.1:8000/admin/

## 🌐 API Reference

The application provides several internal APIs:

- `/suggestions/`: Stock symbol search suggestions
- Stock detail endpoint: View detailed stock information
- News sentiment data

## 📂 Project Structure

```
stockmarketprediction/
├── stocks/                     # Main application
│   ├── management/             # Management commands
│   │   └── commands/           # Custom commands
│   ├── migrations/             # Database migrations
│   ├── templates/              # HTML templates
│   │   └── stocks/             # Stock-specific templates
│   ├── models.py               # Database models
│   ├── views.py                # View controllers
│   ├── services.py             # Business logic
│   ├── news_service.py         # News fetching service
│   └── sentiment_service.py    # Sentiment analysis
├── stockmarketprediction/      # Project settings
├── templates/                  # Global templates
├── static/                     # Static files
├── db.sqlite3                  # SQLite database
└── requirements.txt            # Dependencies
```

## 🔄 Database Schema

The application uses a hybrid database approach:

### SQLite Models
- User authentication
- Stock metadata
- Watchlists and user data
- News articles and sentiment data

### MongoDB Collections
- Historical price data
- News archives
- Market indicators
- Commodity prices

## 🧠 Model Training

StockWise includes a powerful model training system that allows you to train and evaluate multiple predictive models on various stock symbols efficiently.

### Training Models

The training system is located in the `train-model` directory and supports multiple machine learning approaches:

- **XGBoost**: Gradient boosting for regression tasks
- **SVM**: Support Vector Machines for regression
- **LSTM**: Long Short-Term Memory neural networks for sequence prediction
- **ARIMA**: Auto-Regressive Integrated Moving Average for time series forecasting

### Parallel Training Execution

The main training script (`train.py`) orchestrates parallel training jobs with configurable concurrency:

```bash
# Basic usage - train a single model for a single stock
python train-model/train.py --model xgboost --symbol AAPL

# Train multiple model types for a single stock
python train-model/train.py --models xgboost,lstm,arima --symbol MSFT

# Train one model type for multiple stocks
python train-model/train.py --model lstm --symbols AAPL,MSFT,GOOGL

# Train all supported models for multiple stocks
python train-model/train.py --all-models --symbols AAPL,MSFT,GOOGL,AMZN

# Load stock symbols from a file (one symbol per line)
python train-model/train.py --all-models --symbols-file tech_stocks.txt
```

### Training Job Configuration

Control the training process with various command-line options:

| Option | Description |
|--------|-------------|
| `--max-workers` | Maximum number of concurrent training jobs (default: 4) |
| `--model` | Single model type to train (xgboost, svm, lstm, arima) |
| `--models` | Comma-separated list of model types |
| `--all-models` | Train all available model types |
| `--symbol` | Single stock symbol to train on |
| `--symbols` | Comma-separated list of stock symbols |
| `--symbols-file` | Path to a file containing stock symbols (one per line) |
| `--start-date` | Start date for training data (YYYY-MM-DD format) |
| `--end-date` | End date for training data (YYYY-MM-DD format) |
| `--config` | Path to JSON configuration file for batch training jobs |

Additional arguments are passed through to the individual model training scripts.

### Batch Training with JSON Configuration

For complex training scenarios, create a JSON configuration file:

```json
{
  "max_workers": 6,
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "jobs": [
    {
      "model": "xgboost",
      "symbols": ["AAPL", "MSFT", "GOOGL"],
      "extra_args": ["--n-estimators", "300"]
    },
    {
      "model": "lstm",
      "symbols_file": "finance_stocks.txt",
      "extra_args": ["--epochs", "100"]
    },
    {
      "model": "arima",
      "symbol": "TSLA",
      "start_date": "2019-01-01"
    }
  ]
}
```

Run the batch job with:

```bash
python train-model/train.py --config training_jobs.json
```

### Model-Specific Parameters

Each model type supports specific parameters that can be passed as additional arguments:

**XGBoost:**
```bash
python train-model/train.py --model xgboost --symbol AAPL --look-back 60 --features 30 --n-estimators "100,200,300" --max-depth "3,4,5" --learning-rate "0.01,0.1,0.2"
```

The XGBoost model predicts using the following equation:

$$ \hat{y}_i = \sum_{k=1}^K f_k(x_i) $$

where $\hat{y}_i$ is the prediction for the $i$-th instance, $f_k$ represents the $k$-th tree, and $K$ is the total number of trees.

**SVM:**
```bash
python train-model/train.py --model svm --symbol MSFT --look-back 60 --features 30 --kernel rbf --c-values "0.1,1,10,100" --gamma-values "scale,auto,0.1,1,10"
```

**LSTM:**
```bash
python train-model/train.py --model lstm --symbol GOOGL --seq-length 60 --lstm-units 60 --dropout-rate 0.2 --epochs 75 --batch-size 32
```

**ARIMA:**
```bash
python train-model/train.py --model arima --symbol AMZN --price-field close --max-p 3 --max-q 3 --auto-diff
```

### Scheduling Training Jobs

You can schedule regular training jobs using cron (Linux/Mac) or Task Scheduler (Windows):

**Linux/Mac cron example (weekly training):**
```bash
0 2 * * 0 cd /path/to/stockmarketprediction && /path/to/venv/bin/python train-model/train.py --config weekly_training.json >> cron_training.log 2>&1
```

**Monitoring Training Jobs**

Training progress and results are logged to:
- Console output 
- `train_jobs.log` file
- MongoDB evaluation collections (model-specific results)

### Trained Model Storage

Trained models are saved in the following directory structure:
```
train-model/
└── [SYMBOL]/
    └── [MODEL_TYPE]-[TIMESTAMP]/
        ├── model.pkl
        ├── close_scaler.pkl
        ├── other_scaler.pkl
        ├── target_scaler.pkl
        └── selected_features.pkl



## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) for Yahoo Finance data
- [Finnhub](https://finnhub.io/) for financial news API
- [Chart.js](https://www.chartjs.org/) for interactive charts
- [Django](https://www.djangoproject.com/) web framework




gunicorn stockmarketprediction.wsgi:application;

# Navigate to the parent directory containing the dump folder or provide the full path
# Assuming the dump directory 'myDatabaseName' is now at '/home/user/mongodb_backups/myDatabaseName'

mongodump --db stock_data --out ./  

mongorestore /home/user/mongodb_backups/myDatabaseName