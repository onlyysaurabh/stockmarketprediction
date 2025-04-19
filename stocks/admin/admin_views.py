from django.contrib import admin
from django.urls import path
from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils import timezone
from django.contrib.admin.views.decorators import staff_member_required
from django.utils.decorators import method_decorator
from django.views.generic import ListView, DetailView, FormView
from django.urls import reverse_lazy
from ..admin.training_form import ModelTrainingForm
from ..admin.training_manager import TrainingManager
from ..models.training_models import TrainedModel, TrainingJob

class ModelTrainingView(FormView):
    """
    Admin view for training stock prediction models.
    """
    template_name = 'admin/stocks/model_training_form.html'
    form_class = ModelTrainingForm
    success_url = reverse_lazy('admin:stocks_trainedmodel_changelist')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Train Stock Prediction Model'
        context['opts'] = TrainedModel._meta  # For admin breadcrumbs
        
        # Add stock symbol if provided in URL
        if 'stock_id' in self.request.GET:
            context['stock_id'] = self.request.GET.get('stock_id')
            
        return context
    
    def form_valid(self, form):
        # Get form data
        stock = form.cleaned_data['stock']
        model_type = form.cleaned_data['model_type']
        description = form.cleaned_data.get('description', '')
        
        # Get model parameters
        parameters = form.get_model_parameters()
        
        # Generate a name if not provided
        name = form.cleaned_data.get('model_name')
        if not name:
            name = f"{model_type}_{stock.symbol}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get training date range
        start_date, end_date = form.get_date_range()
        
        # Create the TrainedModel record
        trained_model = TrainedModel(
            stock=stock,
            name=name,
            model_type=model_type,
            description=description,
            parameters=parameters,
            status='PENDING',
            training_data_start=start_date,
            training_data_end=end_date,
            forecast_horizon=form.cleaned_data['forecast_horizon'],
            created_by=self.request.user
        )
        trained_model.save()
        
        try:
            # Start the training process
            TrainingManager.train_model(trained_model.id, background=True)
            messages.success(
                self.request,
                f"Training job for {model_type} model on {stock.symbol} started successfully."
            )
        except Exception as e:
            messages.error(
                self.request,
                f"Error starting training job: {str(e)}"
            )
        
        return super().form_valid(form)
    
class TrainedModelListView(ListView):
    """
    Admin view for listing trained models with their status.
    """
    model = TrainedModel
    template_name = 'admin/stocks/trained_model_list.html'
    context_object_name = 'trained_models'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = TrainedModel.objects.all()
        
        # Filter by stock if specified
        stock_id = self.request.GET.get('stock_id')
        if stock_id:
            queryset = queryset.filter(stock_id=stock_id)
            
        # Filter by model type if specified
        model_type = self.request.GET.get('model_type')
        if model_type:
            queryset = queryset.filter(model_type=model_type)
        
        return queryset.order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Trained Models'
        context['opts'] = TrainedModel._meta
        
        # Add filters to context
        context['stock_id'] = self.request.GET.get('stock_id', '')
        context['model_type'] = self.request.GET.get('model_type', '')
        
        return context
    
class TrainedModelDetailView(DetailView):
    """
    Admin view for viewing details of a trained model and its performance.
    """
    model = TrainedModel
    template_name = 'admin/stocks/trained_model_detail.html'
    context_object_name = 'trained_model'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = f'Model: {self.object.name}'
        context['opts'] = TrainedModel._meta
        
        # Add training job information if available
        try:
            context['training_job'] = self.object.training_job
        except:
            context['training_job'] = None
        
        return context
