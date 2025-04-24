from django.db import models
from django.utils import timezone
from datetime import datetime, timedelta

from .models import Stock

class AIAnalysisCache(models.Model):
    """
    Model to store cached AI analysis responses
    """
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='ai_analysis_cache')
    analyzer_type = models.CharField(max_length=20)  # 'gemini', 'vllm', 'groq'
    response_data = models.JSONField()  # Store the full JSON response
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_stale = models.BooleanField(default=False)  # Flag to mark when cache needs refreshing

    class Meta:
        unique_together = ('stock', 'analyzer_type')
        verbose_name_plural = "AI Analysis Cache"

    def __str__(self):
        return f"{self.stock.symbol} AI Analysis ({self.analyzer_type})"

    @property
    def is_fresh(self):
        """Check if the cache is still fresh (less than 24 hours old)"""
        return timezone.now() - self.updated_at < timedelta(hours=24)
