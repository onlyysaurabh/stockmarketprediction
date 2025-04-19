from django import forms
from django.utils import timezone
from datetime import timedelta
from ..models import Stock

class ModelTrainingForm(forms.Form):
    """Form for configuring model training parameters."""
    
    # Stock selection
    stock = forms.ModelChoiceField(
        queryset=Stock.objects.all(),
        required=True,
        label="Stock Symbol",
        help_text="Select the stock to train the model on"
    )
    
    # Model selection
    MODEL_CHOICES = [
        ('SARIMAX', 'SARIMAX - Statistical Time Series'),
        ('LSTM', 'LSTM - Neural Network'),
        ('XGBOOST', 'XGBoost - Gradient Boosting'),
        ('SVM', 'SVM - Support Vector Machine')
    ]
    
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        required=True,
        label="Model Type",
        help_text="Select the type of model to train"
    )
    
    # Time period for training
    PERIOD_CHOICES = [
        ('1Y', '1 Year'),
        ('2Y', '2 Years'),
        ('3Y', '3 Years'),
        ('5Y', '5 Years'),
        ('MAX', 'Maximum Available')
    ]
    
    training_period = forms.ChoiceField(
        choices=PERIOD_CHOICES,
        required=True,
        initial='1Y',
        label="Training Period",
        help_text="Historical data period to use for training"
    )
    
    # Alternative date range selection
    start_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'type': 'date'}),
        label="Start Date",
        help_text="Custom start date for training data"
    )
    
    end_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'type': 'date'}),
        label="End Date",
        help_text="Custom end date for training data"
    )
    
    # Model specific parameters
    # SARIMAX parameters
    auto_parameters = forms.BooleanField(
        required=False,
        initial=True,
        label="Auto-tune Parameters",
        help_text="Automatically find the best parameters for the model"
    )
    
    order_p = forms.IntegerField(
        required=False,
        min_value=0,
        max_value=10,
        initial=1,
        label="AR Order (p)",
        help_text="Autoregressive order"
    )
    
    order_d = forms.IntegerField(
        required=False,
        min_value=0,
        max_value=2,
        initial=1,
        label="Difference Order (d)",
        help_text="Differencing order"
    )
    
    order_q = forms.IntegerField(
        required=False,
        min_value=0,
        max_value=10,
        initial=1,
        label="MA Order (q)",
        help_text="Moving average order"
    )
    
    # LSTM parameters
    sequence_length = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=60,
        initial=10,
        label="Sequence Length",
        help_text="Number of past time steps to use for prediction"
    )
    
    lstm_units = forms.IntegerField(
        required=False,
        min_value=10,
        max_value=200,
        initial=50,
        label="LSTM Units",
        help_text="Number of LSTM units per layer"
    )
    
    lstm_layers = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=5,
        initial=2,
        label="LSTM Layers",
        help_text="Number of LSTM layers"
    )
    
    # XGBoost parameters
    n_estimators = forms.IntegerField(
        required=False,
        min_value=10,
        max_value=1000,
        initial=100,
        label="Number of Estimators",
        help_text="Number of boosting rounds"
    )
    
    max_depth = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=20,
        initial=5,
        label="Max Depth",
        help_text="Maximum depth of trees"
    )
    
    learning_rate = forms.FloatField(
        required=False,
        min_value=0.001,
        max_value=1.0,
        initial=0.1,
        label="Learning Rate",
        help_text="Learning rate for gradient boosting"
    )
    
    # SVM parameters
    KERNEL_CHOICES = [
        ('linear', 'Linear'),
        ('poly', 'Polynomial'),
        ('rbf', 'Radial Basis Function'),
        ('sigmoid', 'Sigmoid')
    ]
    
    kernel = forms.ChoiceField(
        required=False,
        choices=KERNEL_CHOICES,
        initial='rbf',
        label="Kernel",
        help_text="Kernel type for SVM"
    )
    
    c_parameter = forms.FloatField(
        required=False,
        min_value=0.1,
        max_value=100.0,
        initial=1.0,
        label="C Parameter",
        help_text="Regularization parameter"
    )
    
    # Common parameters
    validation_split = forms.FloatField(
        required=False,
        min_value=0.1,
        max_value=0.5,
        initial=0.2,
        label="Validation Split",
        help_text="Proportion of data to use for validation"
    )
    
    forecast_horizon = forms.IntegerField(
        required=True,
        min_value=1,
        max_value=60,
        initial=5,
        label="Forecast Horizon",
        help_text="Number of days to forecast into the future"
    )
    
    model_name = forms.CharField(
        required=False,
        max_length=100,
        label="Model Name",
        help_text="A custom name for this model (optional)"
    )
    
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 3}),
        label="Description",
        help_text="Optional description or notes about this model training"
    )
    
    def clean(self):
        """
        Custom validation to handle conditional fields
        """
        cleaned_data = super().clean()
        model_type = cleaned_data.get("model_type")
        auto_parameters = cleaned_data.get("auto_parameters")
        
        # Validate date range if both provided
        start_date = cleaned_data.get("start_date")
        end_date = cleaned_data.get("end_date")
        if start_date and end_date and start_date > end_date:
            self.add_error('start_date', "Start date must be before end date")
        
        # If auto parameters is turned off, require the parameters for the selected model type
        if not auto_parameters:
            if model_type == 'SARIMAX':
                required_fields = ['order_p', 'order_d', 'order_q']
                for field in required_fields:
                    if not cleaned_data.get(field):
                        self.add_error(field, "This field is required when auto-tuning is off")
            
            elif model_type == 'LSTM':
                required_fields = ['sequence_length', 'lstm_units', 'lstm_layers']
                for field in required_fields:
                    if not cleaned_data.get(field):
                        self.add_error(field, "This field is required")
            
            elif model_type == 'XGBOOST':
                required_fields = ['n_estimators', 'max_depth', 'learning_rate']
                for field in required_fields:
                    if not cleaned_data.get(field):
                        self.add_error(field, "This field is required")
            
            elif model_type == 'SVM':
                required_fields = ['kernel', 'c_parameter']
                for field in required_fields:
                    if not cleaned_data.get(field):
                        self.add_error(field, "This field is required")
        
        return cleaned_data
    
    def get_date_range(self):
        """
        Calculate the start and end dates based on form data
        """
        if self.cleaned_data.get("start_date") and self.cleaned_data.get("end_date"):
            return self.cleaned_data["start_date"], self.cleaned_data["end_date"]
        
        # Calculate date range based on period selection
        period = self.cleaned_data.get("training_period")
        end_date = timezone.now().date()
        
        if period == "1Y":
            start_date = end_date - timedelta(days=365)
        elif period == "2Y":
            start_date = end_date - timedelta(days=730)
        elif period == "3Y":
            start_date = end_date - timedelta(days=1095)
        elif period == "5Y":
            start_date = end_date - timedelta(days=1825)
        else:  # MAX
            start_date = None
        
        return start_date, end_date
    
    def get_model_parameters(self):
        """
        Get parameters specific to the selected model type
        """
        model_type = self.cleaned_data["model_type"]
        auto_parameters = self.cleaned_data.get("auto_parameters", True)
        
        params = {}
        
        if model_type == "SARIMAX":
            if not auto_parameters:
                params = {
                    'order': (
                        self.cleaned_data.get("order_p", 1),
                        self.cleaned_data.get("order_d", 1),
                        self.cleaned_data.get("order_q", 1)
                    )
                }
            else:
                params = {'auto_tune': True}
        
        elif model_type == "LSTM":
            params = {
                'sequence_length': self.cleaned_data.get("sequence_length", 10),
                'units': self.cleaned_data.get("lstm_units", 50),
                'layers': self.cleaned_data.get("lstm_layers", 2),
            }
        
        elif model_type == "XGBOOST":
            params = {
                'n_estimators': self.cleaned_data.get("n_estimators", 100),
                'max_depth': self.cleaned_data.get("max_depth", 5),
                'learning_rate': self.cleaned_data.get("learning_rate", 0.1),
            }
        
        elif model_type == "SVM":
            params = {
                'kernel': self.cleaned_data.get("kernel", 'rbf'),
                'C': self.cleaned_data.get("c_parameter", 1.0),
            }
        
        # Add common parameters
        params['validation_split'] = self.cleaned_data.get("validation_split", 0.2)
        params['horizon'] = self.cleaned_data.get("forecast_horizon", 5)
        
        # Add a custom model name if provided
        if self.cleaned_data.get("model_name"):
            params['model_name'] = self.cleaned_data["model_name"]
        
        return params
