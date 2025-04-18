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
from django.contrib import admin
from django.urls import path, include # Import include
from stocks import admin_views  # Import admin_views for direct URL registration

# Register custom admin URLs before including the default admin URLs
urlpatterns = [
    # Custom admin views must be defined before the admin site URLs
    path('admin/fetch-news/', admin_views.fetch_news_standalone_view, name='admin_fetch_news_standalone'),
    path('admin/stocks/stock/fetch-news/', admin_views.fetch_stock_news_view, name='admin_fetch_stock_news'),
    
    # Prediction model management URLs
    path('admin/predictions/', admin_views.admin_prediction_dashboard, name='admin_prediction_dashboard'),
    path('admin/predictions/train/', admin_views.admin_train_model, name='admin_train_model'),
    path('admin/predictions/model/<int:model_id>/', admin_views.admin_model_detail, name='admin_model_detail'),
    path('admin/predictions/stock/<int:stock_id>/', admin_views.admin_stock_predictions, name='admin_stock_predictions'),
    path('admin/predictions/model/<int:model_id>/delete/', admin_views.admin_delete_model, name='admin_delete_model'),
    
    # Standard admin URLs
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),  # Django auth URLs
    path('', include('stocks.urls')), # Include URLs from the stocks app
]
