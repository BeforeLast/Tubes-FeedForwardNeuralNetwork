from algorithm.ReLU import relu
from algorithm.Softmax import softmax
from algorithm.Linear import linear
from algorithm.Sigmoid import sigmoid

class Neuron:
    """Neuron Class"""
    # Properties
    algorithm:function = None
    weight:list[float] = None

    # Methods
    def __init__(self, algorithm:function, init_weight:list[float], ) -> None:
        """Class constructor"""
        pass

    def calculate(self, input:list[float], ) -> float:
        """Calculate output with the given input"""
        return None
    
    def update(self, ) -> None:
        """Update Neuron's weight"""
        pass