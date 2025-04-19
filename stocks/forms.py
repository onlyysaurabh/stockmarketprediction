from django import forms
from .models import Stock, WatchlistItem
from datetime import date, timedelta
from django.contrib.auth.models import User

class FetchNewsForm(forms.Form):
    """Form for fetching news for selected stocks within a date range."""
    start_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        initial=date.today() - timedelta(days=30),
        help_text='Start date for news retrieval'
    )
    
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        initial=date.today(),
        help_text='End date for news retrieval'
    )
    
    stocks = forms.ModelMultipleChoiceField(
        queryset=Stock.objects.all().order_by('symbol'),
        widget=forms.SelectMultiple(attrs={'class': 'form-control'}),
        required=True,
        help_text='Select stocks to fetch news for'
    )

class TrainModelForm(forms.Form):
    """Form for training a model for stock prediction."""
    stock = forms.ModelChoiceField(
        queryset=Stock.objects.all().order_by('symbol'),
        widget=forms.Select(attrs={'class': 'form-control'}),
        required=True,
        help_text='Select a stock to train the model for'
    )
    
    start_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        initial=date.today() - timedelta(days=365),
        help_text='Start date for training data'
    )
    
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        initial=date.today(),
        help_text='End date for training data'
    )

class ModelTrainingForm(forms.Form):
    MODEL_CHOICES = [
        ('LSTM', 'LSTM Neural Network'),
        ('SARIMA', 'SARIMA Time Series'),
        ('XGBOOST', 'XGBoost'),
        ('SVM', 'Support Vector Machine')
    ]
    
    DURATION_CHOICES = [
        (30, 'Last 30 days'),
        (90, 'Last 90 days'),
        (180, 'Last 180 days'),
        (365, 'Last 365 days'),
        (730, 'Last 2 years'),
        (-1, 'All available data')
    ]
    
    SEQUENCE_LENGTH_CHOICES = [
        (10, '10 days'),
        (20, '20 days'),
        (30, '30 days'),
        (60, '60 days'),
    ]

    models = forms.MultipleChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
        required=True,
        help_text='Select one or more models to train'
    )

    stocks = forms.ModelMultipleChoiceField(
        queryset=Stock.objects.all().order_by('symbol'),
        widget=forms.SelectMultiple(attrs={
            'class': 'form-control',
            'data-placeholder': 'Search and select stocks...'
        }),
        required=True,
        help_text='Search and select one or more stocks for prediction'
    )

    duration = forms.ChoiceField(
        choices=DURATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        required=True,
        help_text='Select the training data duration'
    )
    
    sequence_length = forms.ChoiceField(
        choices=SEQUENCE_LENGTH_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        initial=30,
        required=False,
        help_text='Number of past days to use for sequence models (LSTM)'
    )

class WatchlistItemForm(forms.ModelForm):
    """Form for creating and updating watchlist items."""
    class Meta:
        model = WatchlistItem
        fields = ['symbol', 'tracking_type', 'purchase_price', 
                 'purchase_quantity', 'purchase_date', 'notes']
        widgets = {
            'symbol': forms.TextInput(attrs={'class': 'form-control'}),
            'tracking_type': forms.Select(attrs={'class': 'form-control'}),
            'purchase_price': forms.NumberInput(attrs={'class': 'form-control'}),
            'purchase_quantity': forms.NumberInput(attrs={'class': 'form-control'}),
            'purchase_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }

class UserUpdateForm(forms.ModelForm):
    """Form for updating user profile information."""
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
        }
