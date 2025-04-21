from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import WatchlistItem
from django.utils import timezone

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
        widgets = {
            'symbol': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., AAPL'}),
            'tracking_type': forms.Select(attrs={'class': 'form-control'}),
            'purchase_price': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 150.25', 'step': '0.01'}),
            'purchase_quantity': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 10', 'step': '0.01'}),
            'purchase_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Add any notes about this stock...'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default date to today if adding a new item
        if not self.instance.pk and not self.initial.get('purchase_date'):
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
        widgets = {
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

class TrainModelForm(forms.Form):
    """Form for training model selection and configuration."""
    MODEL_CHOICES = [
        ('xgboost', 'XGBoost'),
        ('lstm', 'LSTM (Deep Learning)'),
        ('svm', 'Support Vector Machine'),
        ('arima', 'ARIMA (Time Series)')
    ]
    
    start_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        label="Training Data Start Date",
        required=True,
        initial='2020-01-01'
    )
    
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        label="Training Data End Date",
        required=True
    )
    
    models = forms.MultipleChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'list-unstyled'}),
        label="Models to Train",
        required=True,
        initial=['xgboost']
    )
    
    max_workers = forms.IntegerField(
        min_value=1,
        max_value=16,
        initial=4,
        label="Maximum Concurrent Jobs",
        help_text="Number of models that can be trained in parallel. Higher values use more CPU/memory.",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default end date to yesterday if not provided
        if not self.initial.get('end_date'):
            self.initial['end_date'] = (timezone.now().date() - timezone.timedelta(days=1))
    
    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get("start_date")
        end_date = cleaned_data.get("end_date")

        if start_date and end_date and end_date < start_date:
            raise forms.ValidationError("End date cannot be before start date.")

        return cleaned_data
