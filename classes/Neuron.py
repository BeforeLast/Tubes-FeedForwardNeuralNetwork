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
        if self.algorithm is None:
            return input
        elif self.algorithm.lower() == "linear":
            return linear(self.weight, input)
        elif self.algorithm.lower() == "relu":
            return relu(self.weight, input)
        elif self.algorithm.lower() == "sigmoid":
            return sigmoid(self.weight, input)
        elif self.algorithm.lower() == "softmax":
            return softmax(input)
        else:  # unknown algorithm
            return input

    # function to set weight to neuron
    def setWeight(self, new_weight: list[float]):
        self.weight = new_weight

    def update(self, ) -> None:
        """Update Neuron's weight"""
        pass

    def __str__(self) -> str:
        result = f"    Neuron: {self.weight}\n"
        return result
