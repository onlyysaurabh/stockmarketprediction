import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
from django.conf import settings
from django.utils import timezone
from ..models.model_factory import ModelFactory
from ..models.training_models import TrainedModel, TrainingJob

class TrainingManager:
    """
    Manager class for handling model training operations.
    """
    
    @staticmethod
    def fetch_stock_data(symbol, start_date=None, end_date=None):
        """
        Fetch historical stock data for the given symbol and date range.
        
        Args:
            symbol (str): Stock symbol
            start_date (datetime.date, optional): Start date for data
            end_date (datetime.date, optional): End date for data
            
        Returns:
            pd.DataFrame: DataFrame with stock data
        """
        try:
            # Use yfinance to download historical data
            stock_data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Ensure the data is not empty
            if stock_data.empty:
                raise ValueError(f"No data available for {symbol} in the specified date range")
                
            return stock_data
            
        except Exception as e:
            raise Exception(f"Failed to fetch data for {symbol}: {str(e)}")
    
    @staticmethod
    def train_model(trained_model_id, background=True):
        """
        Train a model based on the settings in a TrainedModel instance.
        
        Args:
            trained_model_id (int): ID of the TrainedModel to train
            background (bool): Whether to run training in background
            
        Returns:
            dict: Training results or job information
        """
        try:
            # Get the TrainedModel instance
            trained_model = TrainedModel.objects.get(id=trained_model_id)
            
            # Update status to TRAINING
            trained_model.status = 'TRAINING'
            trained_model.save()
            
            # Get or create the training job
            training_job, created = TrainingJob.objects.get_or_create(
                trained_model=trained_model,
                defaults={'is_active': True, 'progress': 0.0}
            )
            
            if not background:
                # Run training in the current thread
                return TrainingManager._execute_training(trained_model, training_job)
            else:
                # In a real implementation, this would use Celery, Django Q, or similar
                # For now, just simulate a background job
                training_job.update_progress(0, "Training job started")
                
                # Here you would start a background task
                # Example: train_model_task.delay(trained_model_id)
                
                return {
                    'job_id': training_job.id,
                    'status': 'PENDING',
                    'message': 'Training job submitted successfully'
                }
                
        except TrainedModel.DoesNotExist:
            raise ValueError(f"TrainedModel with ID {trained_model_id} not found")
        except Exception as e:
            # Handle any other exceptions
            if 'trained_model' in locals():
                trained_model.mark_failed(str(e))
                
            raise Exception(f"Error starting training: {str(e)}")
    
    @staticmethod
    def _execute_training(trained_model, training_job=None):
        """
        Execute the actual model training process.
        
        Args:
            trained_model (TrainedModel): The model to train
            training_job (TrainingJob, optional): Job for tracking progress
            
        Returns:
            dict: Training results
        """
        try:
            # Update progress
            if training_job:
                training_job.update_progress(10, "Fetching data")
            
            # Get stock data
            stock_data = TrainingManager.fetch_stock_data(
                trained_model.stock.symbol,
                trained_model.training_data_start,
                trained_model.training_data_end
            )
            
            # Update progress
            if training_job:
                training_job.update_progress(30, "Data fetched, initializing model")
            
            # Create model instance using the factory
            model = ModelFactory.create_model(
                trained_model.model_type,
                trained_model.stock.symbol,
                model_name=trained_model.name,
                **trained_model.parameters
            )
            
            # Update progress
            if training_job:
                training_job.update_progress(40, "Starting model training")
            
            # Train the model
            train_results = model.train(stock_data)
            
            # Update progress
            if training_job:
                training_job.update_progress(70, "Training complete, evaluating model")
            
            # Evaluate the model (on the most recent data)
            test_size = int(len(stock_data) * 0.2)  # Use the last 20% of data for testing
            test_data = stock_data.iloc[-test_size:]
            evaluation = model.evaluate(test_data)
            
            # Update progress
            if training_job:
                training_job.update_progress(80, "Saving model")
            
            # Create storage directory if it doesn't exist
            model_dir = os.path.join(settings.BASE_DIR, 'model_storage')
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model
            file_path = model.save(directory=model_dir)
            
            # Update the trained model record
            trained_model.status = 'COMPLETED'
            trained_model.file_path = file_path
            trained_model.metrics = evaluation
            trained_model.data_points = len(stock_data)
            trained_model.training_end = timezone.now()
            trained_model.save()
            
            # Complete the training job
            if training_job:
                training_job.update_progress(100, "Training completed successfully")
                training_job.complete()
            
            return {
                'status': 'SUCCESS',
                'file_path': file_path,
                'metrics': evaluation,
                'message': 'Model trained successfully'
            }
            
        except Exception as e:
            error_message = f"Training failed: {str(e)}"
            
            # Update the trained model record
            trained_model.status = 'FAILED'
            trained_model.error_message = error_message
            trained_model.training_end = timezone.now()
            trained_model.save()
            
            # Update the training job
            if training_job:
                training_job.fail(error_message)
            
            return {
                'status': 'FAILED',
                'error': error_message
            }
