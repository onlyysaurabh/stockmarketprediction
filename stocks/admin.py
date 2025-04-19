from django.contrib import admin
from django.contrib import messages
from django.utils.html import format_html
from .models import Watchlist, WatchlistItem, Stock, StockNews
from .services import fetch_and_store_stock_data
import yfinance as yf
from datetime import datetime, date, timedelta
from django.http import HttpResponseRedirect
from django.urls import path, reverse
from django.template.response import TemplateResponse
from django.contrib.admin import SimpleListFilter
from .forms import FetchNewsForm
from django.contrib.admin.models import LogEntry, ADDITION, CHANGE
from django.contrib.contenttypes.models import ContentType
from .admin_utils import fetch_stock_news_view
import json

# Register custom admin site
class StockMarketAdminSite(admin.AdminSite):
    site_header = 'Stock Market Prediction Admin'
    site_title = 'Stock Market Prediction'
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('fetch-stock-news/', 
                 self.admin_view(lambda request: fetch_stock_news_view(request, self)),
                 name='fetch-stock-news'),
        ]
        return custom_urls + urls

admin_site = StockMarketAdminSite(name='stockmarket_admin')

# Custom filter classes for Sector and Industry
class SectorListFilter(SimpleListFilter):
    title = 'Sector'
    parameter_name = 'sector'
    
    def lookups(self, request, model_admin):
        # Get all unique non-null sectors sorted alphabetically
        sectors = Stock.objects.exclude(sector__isnull=True).exclude(sector='').values_list('sector', flat=True).distinct().order_by('sector')
        return [(sector, sector) for sector in sectors]
    
    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(sector=self.value())
        return queryset

class IndustryListFilter(SimpleListFilter):
    title = 'Industry'
    parameter_name = 'industry'
    
    def lookups(self, request, model_admin):
        # Get all unique non-null industries sorted alphabetically
        industries = Stock.objects.exclude(industry__isnull=True).exclude(industry='').values_list('industry', flat=True).distinct().order_by('industry')
        return [(industry, industry) for industry in industries]
    
    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(industry=self.value())
        return queryset

class StockAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'name', 'sector', 'industry', 'get_action_buttons']
    list_filter = [SectorListFilter, IndustryListFilter]
    search_fields = ['symbol', 'name', 'sector', 'industry']

    def get_action_buttons(self, obj):
        update_commodities_url = reverse('admin:update-commodities')
        update_prices_url = reverse('admin:update-all-prices')
        fetch_news_url = reverse('admin:fetch-news')
        return format_html(
            '<a class="button" href="{}">Update Commodities</a>&nbsp;'
            '<a class="button" href="{}">Update All Prices</a>&nbsp;'
            '<a class="button" href="{}">Fetch News</a>',
            update_commodities_url,
            update_prices_url,
            fetch_news_url
        )
    get_action_buttons.short_description = 'Actions'

    def update_all_prices(self, request):
        try:
            # Get all active stocks
            stocks = Stock.objects.filter(is_active=True)
            updated_count = 0
            
            for stock in stocks:
                try:
                    # Use yfinance to fetch latest data
                    ticker = yf.Ticker(stock.symbol)
                    info = ticker.info
                    
                    if info and 'regularMarketPrice' in info:
                        stock.current_price = info['regularMarketPrice']
                        stock.last_updated = datetime.now()
                        stock.save()
                        updated_count += 1
                except Exception as e:
                    messages.warning(request, f"Failed to update {stock.symbol}: {str(e)}")
            
            if updated_count > 0:
                self.message_user(request, f"Successfully updated prices for {updated_count} stocks", messages.SUCCESS)
            else:
                self.message_user(request, "No stock prices were updated", messages.WARNING)
                
        except Exception as e:
            self.message_user(request, f"Error updating stock prices: {str(e)}", messages.ERROR)
        
        return HttpResponseRedirect("../")

    def fetch_news(self, request):
        """
        View for fetching news for stocks
        """
        return fetch_stock_news_view(request, self.admin_site)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('update-commodities/', self.admin_site.admin_view(self.update_commodities), name='update-commodities'),
            path('update-all-prices/', self.admin_site.admin_view(self.update_all_prices), name='update-all-prices'),
            path('fetch-news/', self.admin_site.admin_view(self.fetch_news), name='fetch-news'),
        ]
        return custom_urls + urls

    def update_commodities(self, request):
        try:
            # Get all commodity stocks
            commodities = Stock.objects.filter(sector='Basic Materials')
            for commodity in commodities:
                fetch_and_store_stock_data(commodity.symbol, force_update=True)
            self.message_user(request, "Commodities update completed successfully", messages.SUCCESS)
        except Exception as e:
            self.message_user(request, f"Error updating commodities: {str(e)}", messages.ERROR)
        return HttpResponseRedirect("../")

# Register models with custom admin site
admin_site.register(Stock, StockAdmin)
admin_site.register(StockNews)
admin_site.register(Watchlist)
admin_site.register(WatchlistItem)
