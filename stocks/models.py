from django.db import models
from django.contrib.auth.models import User
# Removed unused: from django.utils import timezone
# Removed unused: from datetime import datetime
# Removed unused: from decimal import Decimal

class Stock(models.Model):
    """
    Stock model to store basic stock information
    """
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    sector = models.CharField(max_length=100, blank=True, null=True)
    industry = models.CharField(max_length=100, blank=True, null=True)
    current_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    previous_close = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    open_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    day_high = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    day_low = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    volume = models.BigIntegerField(null=True, blank=True)
    market_cap = models.BigIntegerField(null=True, blank=True)
    pe_ratio = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    dividend_yield = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    data_updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.symbol}" + (f" - {self.name}" if self.name else "")
    
    class Meta:
        ordering = ['symbol']

class Watchlist(models.Model):
    """
    A watchlist belongs to a user and contains multiple watchlist items (stocks).
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='watchlist')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Watchlist"

class WatchlistItem(models.Model):
    """
    A watchlist item represents a stock in a user's watchlist.
    """
    TRACKING_TYPES = (
        ('TRACKING', 'Just Tracking'),
        ('PURCHASED', 'Purchased'),
    )
    
    watchlist = models.ForeignKey(Watchlist, on_delete=models.CASCADE, related_name='items')
    symbol = models.CharField(max_length=10)
    tracking_type = models.CharField(max_length=10, choices=TRACKING_TYPES, default='TRACKING')
    
    # Fields for purchased stocks
    purchase_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    purchase_quantity = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    purchase_date = models.DateField(null=True, blank=True)
    
    # For tracking current price and calculating returns
    last_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    last_price_check = models.DateTimeField(null=True, blank=True)
    
    # Additional fields
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ('watchlist', 'symbol')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.symbol} ({self.get_tracking_type_display()})"
    
    @property
    def total_investment(self):
        """Calculate the total investment amount."""
        if self.purchase_price and self.purchase_quantity:
            return self.purchase_price * self.purchase_quantity
        return None
    
    @property
    def current_value(self):
        """Calculate the current value of the investment."""
        if self.last_price and self.purchase_quantity:
            return self.last_price * self.purchase_quantity
        return None
    
    @property
    def profit_loss(self):
        """Calculate the profit or loss."""
        if self.current_value is not None and self.total_investment is not None:
            return self.current_value - self.total_investment
        return None
    
    @property
    def profit_loss_percentage(self):
        """Calculate the profit or loss as a percentage."""
        if self.total_investment is not None and self.total_investment > 0 and self.profit_loss is not None:
            return (self.profit_loss / self.total_investment) * 100
        return None
    
    @property
    def percent_change(self):
        """Calculate the percentage change from purchase price to current price."""
        if self.purchase_price and self.last_price and self.purchase_price > 0:
            return ((self.last_price - self.purchase_price) / self.purchase_price) * 100
        return None
    
    def save(self, *args, **kwargs):
        # Only purchased stocks should have purchase details
        if self.tracking_type != 'PURCHASED':
            self.purchase_price = None
            self.purchase_quantity = None
            self.purchase_date = None

        super().save(*args, **kwargs)


class StockNews(models.Model):
    """
    Stores news articles related to a specific stock, including sentiment analysis scores.
    """
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='news')
    source = models.CharField(max_length=50, default='Finnhub') # Source of the news, e.g., Finnhub
    headline = models.TextField()
    summary = models.TextField(blank=True, null=True) # Summary might not always be available
    url = models.URLField(max_length=1024, unique=True) # Use URL as a unique identifier
    published_at = models.DateTimeField() # Timestamp from the source
    sentiment_positive = models.FloatField(null=True, blank=True)
    sentiment_negative = models.FloatField(null=True, blank=True)
    sentiment_neutral = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True) # When the record was created in our DB

    def __str__(self):
        return f"{self.stock.symbol} News: {self.headline[:50]}..."

    class Meta:
        ordering = ['-published_at'] # Show newest news first
        verbose_name_plural = "Stock News"
