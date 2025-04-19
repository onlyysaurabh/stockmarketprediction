from .base_model import BaseModel
from .sarimax_model import SARIMAXModel
from .lstm_model import LSTMModel
from .xgboost_model import XGBoostModel
from .svm_model import SVMModel

class ModelFactory:
    """
    Factory class to create and manage stock prediction models.
    """
    
    @staticmethod
    def get_available_models():
        """
        Get a list of available model types.
        
        Returns:
            list: List of available model names
        """
        return ['SARIMAX', 'LSTM', 'XGBoost', 'SVM']
    
    @staticmethod
    def create_model(model_type, symbol, **kwargs):
        """
        Create a new model instance of the specified type.
        
        Args:
            model_type (str): Type of model to create
            symbol (str): Stock symbol for the model
            **kwargs: Additional parameters for the specific model
            
        Returns:
            BaseModel: An instance of the requested model
            
        Raises:
            ValueError: If the model type is not supported
        """
        model_type = model_type.upper()
        
        if model_type == 'SARIMAX':
            return SARIMAXModel(symbol, **kwargs)
        elif model_type == 'LSTM':
            return LSTMModel(symbol, **kwargs)
        elif model_type == 'XGBOOST':
            return XGBoostModel(symbol, **kwargs)
        elif model_type == 'SVM':
            return SVMModel(symbol, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Available types: {ModelFactory.get_available_models()}")
    
    @staticmethod
    def load_model(model_type, file_path, symbol=None):
        """
        Load a saved model from a file.
        
        Args:
            model_type (str): Type of model to load
            file_path (str): Path to the saved model file
            symbol (str, optional): Stock symbol for the model
            
        Returns:
            BaseModel: The loaded model instance
        """
        # Create a dummy model instance to load from
        model = ModelFactory.create_model(model_type, symbol or "placeholder")
        model.load(file_path)
        return model
