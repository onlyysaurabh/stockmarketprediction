from django.contrib import admin
from django.contrib import messages
from django.utils.html import format_html
from .models import Watchlist, WatchlistItem, Stock
from .services import update_stock_prices
import yfinance as yf
from datetime import datetime
from django.http import HttpResponseRedirect
from django.urls import path
from django.template.response import TemplateResponse
from django.contrib.admin import SimpleListFilter

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

# Register your models here.

@admin.register(Watchlist)
class WatchlistAdmin(admin.ModelAdmin):
    list_display = ('user', 'created_at')
    search_fields = ('user__username',)

@admin.register(WatchlistItem)
class WatchlistItemAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'watchlist_user', 'tracking_type', 'purchase_price', 'purchase_quantity', 'created_at')
    list_filter = ('tracking_type', 'watchlist__user')
    search_fields = ('symbol', 'watchlist__user__username')
    readonly_fields = ('created_at',) # Using created_at instead of added_at

    # Helper method to display the user from the related Watchlist
    def watchlist_user(self, obj):
        return obj.watchlist.user
    watchlist_user.short_description = 'User' # Column header

@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'name', 'get_current_price', 'get_price_change', 'sector', 'industry', 'data_updated')
    list_filter = (SectorListFilter, IndustryListFilter)  # Use custom filter classes
    search_fields = ('symbol', 'name', 'sector', 'industry')
    readonly_fields = ('data_updated',)
    actions = ['update_selected_stock_prices']
    list_per_page = 50  # Show more stocks per page
    list_max_show_all = 1000  # Allow showing up to 1000 stocks when clicking "Show All"
    show_full_result_count = True  # Show the total count of stocks

    def get_current_price(self, obj):
        if obj.current_price:
            return f"${obj.current_price}"
        return "-"
    get_current_price.short_description = "Current Price"
    
    def get_price_change(self, obj):
        if obj.current_price and obj.previous_close and obj.previous_close > 0:
            change = obj.current_price - obj.previous_close
            change_percent = (change / obj.previous_close) * 100
            color = 'green' if change >= 0 else 'red'
            arrow = '▲' if change >= 0 else '▼'
            # Convert values to strings before using format_html to avoid SafeString formatting issues
            change_str = f"{float(change):.2f}"
            percent_str = f"{float(change_percent):.2f}"
            return format_html('<span style="color: {};">{} ${} ({}%)</span>', 
                            color, arrow, change_str, percent_str)
        return "-"
    get_price_change.short_description = "Price Change"

    def update_selected_stock_prices(self, request, queryset):
        """Admin action to update prices for selected stocks using the robust update mechanism"""
        symbols = list(queryset.values_list('symbol', flat=True))
        
        # Call our robust update function
        results = update_stock_prices(symbols=symbols, delay=1.0)
        
        # Show messages based on results
        if results['success'] > 0:
            self.message_user(
                request, 
                f"Successfully updated {results['success']} stock(s).", 
                level=messages.SUCCESS
            )
        
        if results['not_found'] > 0:
            self.message_user(
                request,
                f"Could not find data for {results['not_found']} stock(s).",
                level=messages.WARNING
            )
            
        if results['failed'] > 0:
            self.message_user(
                request,
                f"Failed to update {results['failed']} stock(s) due to errors.",
                level=messages.ERROR
            )
            
    update_selected_stock_prices.short_description = "Update prices for selected stocks"
    
    def get_actions(self, request):
        actions = super().get_actions(request)
        return actions

    def get_urls(self):
        """Add custom URL patterns for the admin interface"""
        urls = super().get_urls()
        custom_urls = [
            path('update-all-prices/', self.admin_site.admin_view(self.update_all_prices_view), name='stocks_stock_update_all_prices'),
        ]
        return custom_urls + urls
    
    def update_all_prices_view(self, request):
        """Custom admin view to update all stock prices using our robust update mechanism"""
        if request.method == 'POST':
            # Use our robust update_stock_prices function
            results = update_stock_prices(delay=1.0)
            
            # Show summary messages
            if results['success'] > 0:
                self.message_user(
                    request, 
                    f"Successfully updated {results['success']} stock(s).", 
                    level=messages.SUCCESS
                )
            
            if results['not_found'] > 0:
                self.message_user(
                    request,
                    f"Could not find data for {results['not_found']} stock(s).",
                    level=messages.WARNING
                )
                
            if results['failed'] > 0:
                self.message_user(
                    request,
                    f"Failed to update {results['failed']} stock(s) due to errors.",
                    level=messages.ERROR
                )
                
                # Show the first few errors
                error_count = 0
                for symbol, error in results['errors'].items():
                    if error_count < 5:  # Show at most 5 detailed errors
                        self.message_user(
                            request,
                            f"Error updating {symbol}: {error}",
                            level=messages.ERROR
                        )
                        error_count += 1
                
                if error_count < len(results['errors']):
                    self.message_user(
                        request,
                        f"... and {len(results['errors']) - error_count} more errors",
                        level=messages.ERROR
                    )
            
            return HttpResponseRedirect("../")
        
        # If not POST, show confirmation page
        context = {
            'title': 'Confirm Price Update',
            'stock_count': Stock.objects.count(),
            'opts': self.model._meta,
        }
        return TemplateResponse(request, "admin/stocks/stock/update_all_prices.html", context)
        
    def changelist_view(self, request, extra_context=None):
        """Override to add the update button to the template"""
        extra_context = extra_context or {}
        extra_context['show_update_button'] = True
        return super().changelist_view(request, extra_context=extra_context)

    def get_list_display_links(self, request, list_display):
        # Make symbol and name clickable for easy navigation to detail view
        return ('symbol', 'name')
    
    def get_search_results(self, request, queryset, search_term):
        # Enhanced search to prioritize symbol matches and improve full-text search
        queryset, use_distinct = super().get_search_results(request, queryset, search_term)
        
        # If searching by symbol, prioritize exact matches
        if search_term:
            # Look for exact symbol matches first
            exact_symbol_matches = self.model.objects.filter(symbol__iexact=search_term)
            
            # Then try partial symbol matches
            partial_symbol_matches = self.model.objects.filter(
                symbol__icontains=search_term
            ).exclude(pk__in=exact_symbol_matches.values_list('pk', flat=True))
            
            # Then try exact name matches
            exact_name_matches = self.model.objects.filter(
                name__iexact=search_term
            ).exclude(
                pk__in=exact_symbol_matches.values_list('pk', flat=True)
            ).exclude(
                pk__in=partial_symbol_matches.values_list('pk', flat=True)
            )
            
            # Combine all matches with exact matches first
            if exact_symbol_matches.exists() or partial_symbol_matches.exists() or exact_name_matches.exists():
                combined_pks = set()
                result_queryset = []
                
                # Add exact symbol matches
                for item in exact_symbol_matches:
                    if item.pk not in combined_pks:
                        result_queryset.append(item)
                        combined_pks.add(item.pk)
                
                # Add partial symbol matches
                for item in partial_symbol_matches:
                    if item.pk not in combined_pks:
                        result_queryset.append(item)
                        combined_pks.add(item.pk)
                
                # Add exact name matches
                for item in exact_name_matches:
                    if item.pk not in combined_pks:
                        result_queryset.append(item)
                        combined_pks.add(item.pk)
                
                # Add any remaining matches from the original queryset
                for item in queryset:
                    if item.pk not in combined_pks:
                        result_queryset.append(item)
                        combined_pks.add(item.pk)
                
                # Return the combined queryset with our priority ordering
                return result_queryset, use_distinct
                
        return queryset, use_distinct
