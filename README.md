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

## 📝 License

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