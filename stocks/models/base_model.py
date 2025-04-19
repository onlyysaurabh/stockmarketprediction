from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import os

class BaseModel(ABC):
    """
    Abstract base class for all stock prediction models.
    This provides common functionality and enforces a consistent interface.
    """
    
    def __init__(self, symbol, model_name=None):
        """
        Initialize the base model.
        
        Args:
            symbol (str): The stock symbol this model will be trained on
            model_name (str, optional): Custom name for this model instance
        """
        self.symbol = symbol
        self.model_name = model_name or f"{self.__class__.__name__}_{symbol}"
        self.model = None
        self.trained = False
        self.metadata = {
            'symbol': symbol,
            'model_type': self.__class__.__name__,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'training_parameters': {},
            'performance_metrics': {},
        }
    
    @abstractmethod
    def preprocess_data(self, data):
        """
        Preprocess input data for training or prediction.
        
        Args:
            data (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Processed data ready for model consumption
        """
        pass
    
    @abstractmethod
    def train(self, data, **kwargs):
        """
        Train the model on the provided data.
        
        Args:
            data (pd.DataFrame): Historical stock data
            **kwargs: Additional training parameters
            
        Returns:
            dict: Training results and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data, horizon=5, **kwargs):
        """
        Generate predictions using the trained model.
        
        Args:
            data (pd.DataFrame): Input data for prediction
            horizon (int): Number of days to predict into the future
            **kwargs: Additional prediction parameters
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data):
        """
        Evaluate model performance on test data.
        
        Args:
            test_data (pd.DataFrame): Test data with actual values
            
        Returns:
            dict: Performance metrics
        """
        pass
    
    def save(self, directory="model_storage"):
        """
        Save the trained model to disk.
        
        Args:
            directory (str): Directory to save the model
            
        Returns:
            str: Path to the saved model file
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        self.metadata['updated_at'] = datetime.now().isoformat()
        file_path = os.path.join(directory, f"{self.model_name}.pkl")
        
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metadata': self.metadata
            }, f)
            
        return file_path
    
    def load(self, file_path):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str): Path to the saved model file
            
        Returns:
            bool: True if loaded successfully
        """
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
            
        self.model = loaded_data['model']
        self.metadata = loaded_data['metadata']
        self.trained = True
        
        return True
    
    def get_hyperparameters(self):
        """
        Get the current model hyperparameters.
        
        Returns:
            dict: Model hyperparameters
        """
        return self.metadata.get('training_parameters', {})
    
    def set_hyperparameters(self, **kwargs):
        """
        Set model hyperparameters.
        
        Args:
            **kwargs: Hyperparameters to set
            
        Returns:
            None
        """
        self.metadata['training_parameters'].update(kwargs)
