# StockWise: Stock Market Prediction & Analysis

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
# Django Configuration
SECRET_KEY=your-secret-key-here

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=stock_data

# Finnhub API Keys (for news data)
FINNHUB_API_KEYS=your_api_key_here
```

### 🔑 API Keys Configuration

This application requires API keys for external data services. Here's where and how to configure them:

#### **Finnhub API Keys (Required for News Data)**

1. **Get API Keys:**
   - Visit [Finnhub.io](https://finnhub.io/)
   - Sign up for a free account
   - Navigate to your dashboard to get your API key

2. **Add to .env file:**
   ```
   FINNHUB_API_KEYS=your_actual_api_key_here
   ```
   
   **For multiple API keys (recommended for higher rate limits):**
   ```
   FINNHUB_API_KEYS=key1,key2,key3
   ```

3. **Location:** The `.env` file should be in the project root directory (same level as `manage.py`)

#### **Django Secret Key (Required)**

1. **Generate a new secret key:**
   ```python
   from django.core.management.utils import get_random_secret_key
   print(get_random_secret_key())
   ```

2. **Add to .env file:**
   ```
   SECRET_KEY=your-generated-secret-key-here
   ```

#### **MongoDB Configuration (Optional)**

If using a remote MongoDB instance:
```
MONGO_URI=mongodb://username:password@host:port/
MONGO_DB_NAME=your_database_name
```

**⚠️ Security Note:** Never commit your `.env` file to version control. It's already included in `.gitignore`.

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

## � Troubleshooting

### Common API Key Issues

**Problem: "No Finnhub API keys found" error**
- **Solution:** Ensure your `.env` file is in the project root directory (same level as `manage.py`)
- **Check:** Verify the variable name is exactly `FINNHUB_API_KEYS` (case-sensitive)
- **Test:** Run `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('FINNHUB_API_KEYS'))"` to verify

**Problem: "Invalid API key" error**
- **Solution:** Double-check your API key from your Finnhub dashboard
- **Note:** Free accounts have rate limits (60 calls/minute)

**Problem: News data not updating**
- **Check:** API key permissions on Finnhub dashboard
- **Solution:** Try rotating API keys if you have multiple

**Problem: Django "SECRET_KEY" errors**
- **Solution:** Generate a new secret key and add it to your `.env` file
- **Command:** `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`

### Environment File Location
```
stockmarketprediction/          ← Project root
├── .env                        ← Your .env file goes here
├── manage.py
├── requirements.txt
└── stockmarketprediction/
    └── settings.py
```

### 🚨 Already Committed API Keys to Git?

If you accidentally committed API keys to Git history, see the detailed guide: [`REMOVE_API_KEYS_GUIDE.md`](./REMOVE_API_KEYS_GUIDE.md)

**Quick emergency steps:**
1. **Immediately revoke the exposed API keys** from your provider dashboards
2. **Generate new API keys** before cleaning Git history
3. **Follow the complete cleanup guide** to remove from Git/GitHub history

## �📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Lead Developer: Your Name
- Data Scientist: Contributor Name
- UX/UI Designer: Designer Name

## 🙏 Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) for Yahoo Finance data
- [Finnhub](https://finnhub.io/) for financial news API
- [Chart.js](https://www.chartjs.org/) for interactive charts
- [Django](https://www.djangoproject.com/) web framework