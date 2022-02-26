from classes.Neuron import Neuron


class Layer:
    """Layer Class"""
    # Properties
    alogrithm:function = None
    neurons:list[Neuron] = None
    trainable:bool = None
    name:str = None

    # Methods
    def __init__(self,):
        """Class constructor"""
        pass
    
    def calculate(self, input:list[float]) -> list[float]:
        """Calculate layer output from the given input"""
        return []
    
    def update(self,) -> None:
        """Update each neurons"""
        pass
    
