from django.db import models
from django.utils import timezone
from django.conf import settings
import os
import json
from ..models import Stock

class TrainedModel(models.Model):
    """Model for storing trained prediction model information."""
    
    MODEL_TYPES = [
        ('SARIMAX', 'SARIMAX - Statistical Time Series'),
        ('LSTM', 'LSTM - Neural Network'),
        ('XGBOOST', 'XGBoost - Gradient Boosting'),
        ('SVM', 'SVM - Support Vector Machine')
    ]
    
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('TRAINING', 'Training'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed')
    ]
    
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='trained_models')
    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    description = models.TextField(blank=True, null=True)
    file_path = models.CharField(max_length=255, blank=True, null=True)
    parameters = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    training_start = models.DateTimeField(default=timezone.now)
    training_end = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    error_message = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.SET_NULL, 
        null=True, 
        related_name='trained_models'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Training data information
    training_data_start = models.DateField(null=True, blank=True) 
    training_data_end = models.DateField(null=True, blank=True)
    data_points = models.IntegerField(null=True, blank=True)
    
    # Prediction horizon
    forecast_horizon = models.IntegerField(default=5)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Trained Model"
        verbose_name_plural = "Trained Models"
    
    def __str__(self):
        return f"{self.name} - {self.stock.symbol} ({self.model_type})"
    
    @property
    def model_file_exists(self):
        """Check if the model file exists on disk."""
        if not self.file_path:
            return False
        return os.path.exists(self.file_path)
    
    @property
    def training_duration(self):
        """Calculate the duration of the training process."""
        if self.training_end and self.training_start:
            duration = self.training_end - self.training_start
            return duration.total_seconds()
        return None
    
    def mark_completed(self, file_path, metrics):
        """Mark the model training as completed."""
        self.status = 'COMPLETED'
        self.file_path = file_path
        self.metrics = metrics
        self.training_end = timezone.now()
        self.save()
    
    def mark_failed(self, error_message):
        """Mark the model training as failed."""
        self.status = 'FAILED'
        self.error_message = error_message
        self.training_end = timezone.now()
        self.save()
    
    def get_metrics_display(self):
        """Format metrics for display."""
        if not self.metrics:
            return "No metrics available"
        
        result = []
        for key, value in self.metrics.items():
            if isinstance(value, (int, float)):
                if key.lower() in ['mape', 'r2']:
                    result.append(f"{key.upper()}: {value:.2f}%")
                else:
                    result.append(f"{key.upper()}: {value:.4f}")
        
        return ", ".join(result)
    
    def get_parameters_display(self):
        """Format parameters for display."""
        if not self.parameters:
            return "Default parameters"
        
        result = []
        # Filter out less important parameters for display
        important_params = ['order', 'sequence_length', 'units', 'layers', 
                         'n_estimators', 'max_depth', 'learning_rate', 'kernel', 'C']
        
        for key, value in self.parameters.items():
            if key in important_params:
                result.append(f"{key}: {value}")
        
        return ", ".join(result)
    
class TrainingJob(models.Model):
    """Model for tracking asynchronous training jobs."""
    
    trained_model = models.OneToOneField(
        TrainedModel, 
        on_delete=models.CASCADE, 
        related_name='training_job'
    )
    job_id = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    progress = models.FloatField(default=0.0)  # 0-100%
    log_output = models.TextField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Training job for {self.trained_model}"
    
    def update_progress(self, progress, log_line=None):
        """Update the job progress and log."""
        self.progress = min(max(progress, 0), 100)  # Ensure progress is between 0-100
        
        if log_line:
            if self.log_output:
                self.log_output = f"{self.log_output}\n{log_line}"
            else:
                self.log_output = log_line
        
        self.save()
    
    def complete(self):
        """Mark the job as complete."""
        self.progress = 100
        self.is_active = False
        self.save()
    
    def fail(self, error_message):
        """Mark the job as failed."""
        self.is_active = False
        self.update_progress(self.progress, f"ERROR: {error_message}")
        self.save()
