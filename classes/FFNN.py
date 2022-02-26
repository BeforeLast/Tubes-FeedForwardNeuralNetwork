from typing import Any
from classes.Layer import Layer

class FFNN:
    """Feed Forward Neural Network: class for Feed Forward
    Neural Network Model"""
    # Properties
    name:str = None
    layers:list = None
    learning_rate:float = None

    # Methods
    def __init__(self, file_path:str="model.json") -> None:
        """Class constructor"""
        pass

    def train(self, input, epoch:int = 10, ) -> None:
        """Train model with given input"""
        pass

    def predict(self, input) -> Any:
        """Predict output from given input"""
        pass

    def visualize(self) -> Any:
        """Visualize FFNN model"""
        pass

    def load(self, file_path:str="model.json", ) -> None:
        """Load model from external file"""
        pass
    
    def save(self, file_path:str="model.json", ) -> None:
        """Save model to external file"""
        pass
    