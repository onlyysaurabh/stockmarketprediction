from django.urls import path, re_path
from . import views
from . import prediction_views
# Removed admin_views import as we're now registering these URLs at the project level
# Removed unused: from django.contrib.auth.views import LogoutView

# Define URL patterns for the stocks app
# Use 'name' for easy URL reversing in templates and views
urlpatterns = [
    # Example: / (Homepage with search)
    path('', views.home, name='home'),

    # Example: /suggestions/?term=AAP (GET endpoint for search suggestions)
    path('suggestions/', views.search_suggestions_view, name='search_suggestions'),

    # User authentication
    path('register/', views.register_view, name='register'),
    path('account/', views.account_view, name='account'), # Add account page URL
    path('logout/', views.logout_view, name='logout'),  # Use our custom logout view

    # Watchlist URLs
    path('watchlist/', views.watchlist, name='watchlist'),
    path('watchlist/add/', views.add_to_watchlist, name='add_to_watchlist'),
    path('watchlist/add/<str:symbol>/', views.add_to_watchlist_with_symbol, name='add_to_watchlist_with_symbol'),
    path('watchlist/edit/<int:item_id>/', views.edit_watchlist_item, name='edit_watchlist_item'),
    path('watchlist/remove/<int:item_id>/', views.remove_from_watchlist, name='remove_from_watchlist'),

    # News URLs
    path('news/', views.stock_news, name='stock_news'),  # News page without symbol
    path('news/<str:symbol>/', views.stock_news, name='stock_news_with_symbol'),  # News page with symbol
    
    # Prediction URLs
    path('predictions/<str:symbol>/', prediction_views.stock_predictions, name='stock_predictions'),
    path('models/comparison/', prediction_views.model_comparison, name='model_comparison'),

    # Example: /AAPL/update/ (POST endpoint to trigger data update)
    path('<str:symbol>/update/', views.update_stock_data, name='update_stock_data'),

    # Example: /AAPL/ (Stock detail page)
    path('<str:symbol>/', views.stock_detail, name='stock_detail'),

    # Removed custom admin URLs as they're now registered at the project level
]
