from algorithm.ReLU import relu
from algorithm.Softmax import softmax
from algorithm.Linear import linear
from algorithm.Sigmoid import sigmoid

class Neuron:
    """Neuron Class"""
    # Properties
    algorithm:str = None
    weight:list[float] = None
    bias:float = None

    # Methods
    def __init__(self, algorithm:str, init_weight:list[float],bias:float ) -> None:
        """Class constructor"""
        self.algorithm = algorithm
        self.weight = init_weight
        self.bias = bias

    def calculate(self, input:list[float], ) -> float:
        """Calculate output with the given input"""
        if self.algorithm == "linear":
            return linear(self.weight,input,self.bias)
        elif self.algorithm == "relu":
            return relu(self.weight,input, self.bias)
        elif self.algorithm == "sigmoid":
            return sigmoid(self.weight, input, self.bias)
        else: #algo == softmax
            #- softmax implementation belum
            return linear(self.weight,input,self.bias)

    #function to set weight to neuron 
    def set_weight(self, new_weight:list[float]):
        self.weight = new_weight

    def update(self, ) -> None:
        """Update Neuron's weight"""
        pass