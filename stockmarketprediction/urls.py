"""stockmarketprediction URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from typing import List # Import List for type hinting
from django.contrib import admin
# Import URLPattern and URLResolver for type hinting urlpatterns
from django.urls import path, include, URLPattern, URLResolver
from stocks import admin_views  # Import admin_views for direct URL registration

# Register custom admin URLs before including the default admin URLs
# Add type hint for urlpatterns
urlpatterns: List[URLPattern | URLResolver] = [
    # Custom admin views must be defined before the admin site URLs
    path('admin/fetch-news/', admin_views.fetch_news_standalone_view, name='admin_fetch_news_standalone'),
    path('admin/stocks/stock/fetch-news/', admin_views.fetch_stock_news_view, name='admin_fetch_stock_news'),
    path('admin/train-model/', admin_views.train_model_view, name='admin:train_model'),  # Updated URL name with admin namespace
    
    # Standard admin URLs
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),  # Django auth URLs
    path('', include('stocks.urls')), # Include URLs from the stocks app
]
