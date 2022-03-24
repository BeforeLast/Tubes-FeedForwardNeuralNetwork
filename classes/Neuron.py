from unittest import case
from algorithm.Error import softmaxError
from algorithm.Gradient import derLinear, derRelu, derSigmoid, derSoftmax
from algorithm.ReLU import relu
from algorithm.Softmax import softmax
from algorithm.Linear import linear
from algorithm.Sigmoid import sigmoid


class Neuron:
    """Neuron Class"""
    # Properties
    algorithm: str = None
    weight: list[float] = None

    # Methods
    def __init__(self, algorithm: str, init_weight: list[float], ) -> None:
        """Class constructor"""
        self.algorithm = algorithm
        self.weight = init_weight

    def calculate(self, input: list[float], ) -> float:
        """Calculate output with the given input"""
        if self.algorithm is None: #untuk input layer
            return input
        elif self.algorithm.lower() == "linear":
            return linear(self.weight, input)
        elif self.algorithm.lower() == "relu":
            return relu(self.weight, input)
        elif self.algorithm.lower() == "sigmoid":
            return sigmoid(self.weight, input)
        else:  # unknown algorithm or softmax (softmax juga cuma dipake di output layer, so just return input)
            return input

    # function to set weight to neuron
    def setWeight(self, new_weight: list[float]):
        """Set Neuron's weight"""
        self.weight = new_weight
    
    def getWeight(self) -> list[float]:
        """Return copy of neuron's weight"""
        return self.weight.copy()
    
    def update(self, delta_weight: list[float]) -> None:
        """Update Neuron's weight"""
        for i in range (len(self.weight)):
            self.weight[i] += delta_weight[i]

    def errorTerm(self,
        input: float = None, output: float = None, label: float = None,
        output_neuron: bool = None, sigma: float = None,
        softmax: float = None):
        """Calculate neuron's errorterm"""
        if output_neuron:
            # Errorterm if neuron is at output layer
            if self.algorithm is None: #untuk input layer
                return input
            elif self.algorithm.lower() == "linear":
                return derLinear(output) * (output-label)
            elif self.algorithm.lower() == "relu":
                return derRelu(output) * (output-label)
            elif self.algorithm.lower() == "sigmoid":
                return derSigmoid(output) * (output-label)
            else: # TODO: softmax
                return 0
        else:
            # Errorterm if neuron is not at hidden layer
            if self.algorithm is None: #untuk input layer
                return input
            elif self.algorithm.lower() == "linear":
                return derLinear(output) * sigma
            elif self.algorithm.lower() == "relu":
                return derRelu(output) * sigma
            elif self.algorithm.lower() == "sigmoid":
                return derSigmoid(output) * sigma
            else: # TODO: softmax
                return 0

    def __str__(self) -> str:
        """Return class as string"""
        result = f"    Neuron: {self.weight}\n"
        return result
