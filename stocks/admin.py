from django.contrib import admin
from .models import Watchlist, WatchlistItem # Import your models

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
