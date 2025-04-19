from typing import Any, Optional, Dict # Reorder imports, Dict is used now
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import WatchlistItem, Stock # Import Stock model
from django.utils import timezone
from django.db.models.query import QuerySet # Import QuerySet for type hinting

class RegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

class WatchlistItemForm(forms.ModelForm):
    """Form for adding and editing watchlist items."""
    
    class Meta:
        model = WatchlistItem
        fields = ['symbol', 'tracking_type', 'purchase_price', 'purchase_quantity', 'purchase_date', 'notes']
        # Add type hint for widgets dictionary
        widgets: Dict[str, forms.Widget] = {
            'symbol': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., AAPL'}),
            'tracking_type': forms.Select(attrs={'class': 'form-control'}),
            'purchase_price': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 150.25', 'step': '0.01'}),
            'purchase_quantity': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 10', 'step': '0.01'}),
            'purchase_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Add any notes about this stock...'}),
        }

    # Add type hints for *args and **kwargs
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Set default date to today if adding a new item
        if not self.instance.pk and not self.initial.get('purchase_date'): # type: ignore Pylance reports instance might not exist, but it does on ModelForm
            self.initial['purchase_date'] = timezone.now().date()
            
    def clean(self):
        cleaned_data = super().clean()
        tracking_type = cleaned_data.get('tracking_type')
        purchase_price = cleaned_data.get('purchase_price')
        purchase_quantity = cleaned_data.get('purchase_quantity')
        purchase_date = cleaned_data.get('purchase_date')
        
        # Validate that purchase details are provided for purchased stocks
        if tracking_type == 'PURCHASED':
            if not purchase_price:
                self.add_error('purchase_price', 'Purchase price is required for purchased stocks.')
            if not purchase_quantity:
                self.add_error('purchase_quantity', 'Quantity is required for purchased stocks.')
            if not purchase_date:
                self.add_error('purchase_date', 'Purchase date is required for purchased stocks.')
        
        return cleaned_data

class UserUpdateForm(forms.ModelForm):
    """Form for updating user profile information (excluding password)."""
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name']
        # Add type hint for widgets dictionary
        widgets: Dict[str, forms.Widget] = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
        }

    def clean_email(self):
        """Ensure the email address is unique."""
        email = self.cleaned_data.get('email')
        username = self.cleaned_data.get('username')
        if email and User.objects.filter(email=email).exclude(username=username).exists():
            raise forms.ValidationError('This email address is already in use. Please use another.')
        return email


class FetchNewsForm(forms.Form):
    """Form for selecting date range for fetching news in admin."""
    start_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        label="Start Date",
        required=True
    )
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        label="End Date",
        required=True
    )

    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get("start_date")
        end_date = cleaned_data.get("end_date")

        if start_date and end_date and end_date < start_date:
            raise forms.ValidationError("End date cannot be before start date.")

        return cleaned_data


# --- Model Training Form ---

class ModelTrainingForm(forms.Form):
    """Form for configuring and initiating model training."""
    MODEL_CHOICES = [
        ('LSTM', 'LSTM (Long Short-Term Memory)'),
        ('GRU', 'GRU (Gated Recurrent Unit)'),
        # Add other model types here as they become available
    ]

    # Use ModelMultipleChoiceField for selecting stocks
    # We'll populate the queryset in the view
    # Initialize with an empty queryset of the correct type
    stocks = forms.ModelMultipleChoiceField(
        queryset=Stock.objects.none(), # Initialize with empty Stock queryset
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'stock-checkbox-list'}),
        label="Select Stocks to Train On",
        required=True
    )

    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label="Select Model Type",
        required=True
    )

    start_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        label="Training Data Start Date",
        required=True
    )
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        label="Training Data End Date",
        required=True
    )

    epochs = forms.IntegerField(
        min_value=1,
        initial=50,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 50'}),
        label="Number of Epochs",
        required=True
    )

    batch_size = forms.IntegerField(
        min_value=1,
        initial=32,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 32'}),
        label="Batch Size",
        required=True
    )

    # Add more hyperparameter fields as needed (e.g., learning rate, sequence length)

    # Add type hints for *args and **kwargs, and the specific stock_queryset kwarg
    def __init__(self, *args: Any, stock_queryset: Optional[QuerySet[Stock]] = None, **kwargs: Any):
        # No need to pop, just pass other kwargs to super
        super().__init__(*args, **kwargs)
        # Set the queryset for the stocks field if provided
        if stock_queryset is not None:
            # Ensure the field exists and is of the correct type before assigning queryset
            stocks_field = self.fields.get('stocks')
            if isinstance(stocks_field, forms.ModelMultipleChoiceField):
                 stocks_field.queryset = stock_queryset
            else:
                 # Handle case where 'stocks' field might not be what we expect
                 # This shouldn't happen with the current form definition but is safer
                 pass

    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get("start_date")
        end_date = cleaned_data.get("end_date")

        if start_date and end_date:
            if end_date < start_date:
                raise forms.ValidationError("End date cannot be before start date.")
            # Optional: Add validation for minimum training data duration

        # Optional: Add validation for stock selection (e.g., at least one selected)
        if not cleaned_data.get('stocks'):
             # This might be redundant due to required=True, but good practice
             raise forms.ValidationError("Please select at least one stock.")

        return cleaned_data
