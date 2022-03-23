from unittest import case
from algorithm.Error import error, softmaxError
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

    def update(self, output_error_array, self_output, learn_rate) -> None:
        """Update Neuron's weight"""
        # calculate hidden error
        hiddenError = self_output*(1-self_output)*linear(self.weight,output_error_array)
        # find gradient
        if self.algorithm.lower() == "linear":
            grad = derLinear(self_output)
        elif self.algorithm.lower() == "relu":
            grad = derRelu(self_output)
        elif self.algorithm.lower() == "sigmoid":
            grad = derSigmoid(self_output)
        else: # softmax masuk sini dulu i think, buat output layer harusnya kan ga update weight
            grad = derLinear(self_output)

        grad = grad*-1*learn_rate

        # update weight
        # pls cmiiw, deltaweight = grad * hiddenError * currentweight
        for i in range (len(self.weight)):
            self.weight[i] = self.weight[i] + grad * hiddenError * self.weight[i]

    def getHiddenError(self, output_error_array, self_output):
        hiddenError = self_output*(1-self_output)*linear(self.weight,output_error_array)
        return hiddenError

    def __str__(self) -> str:
        """Return class as string"""
        result = f"    Neuron: {self.weight}\n"
        return result
