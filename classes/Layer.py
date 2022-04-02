from __future__ import annotations
from algorithm.Softmax import softmax
from classes.Neuron import Neuron

class Layer:
    """Layer Class"""
    # Properties
    algorithm: str = None
    neurons: list[Neuron] = None
    trainable: bool = None
    name: str = None
    layer_bias: float = None
    input_history: list[float] = []
    output_history: list[float] = []
    errorterm_history: list[float] = []

    # Methods
    def __init__(self, algorithm: str, n_neurons: int,
                 trainable: bool, name: str, layer_bias: float):
        """Class constructor"""
        # create a layer with n neurons, each having [] as
        # initial_weight (empty array)
        # akan diasumsi weight setting untuk either setiap neuron or
        # individual neuron akan dilakukan di model
        self.algorithm = algorithm
        self.trainable = trainable
        self.name = name
        self.layer_bias = layer_bias
        self.neurons = []
        for _ in range(n_neurons):
            self.neurons.append(Neuron(algorithm, []))

    def getAlgorithm(self) -> str:
        """Return layer's algorithm"""
        return self.algorithm

    # getter setter neuron list
    def getNeuronList(self) -> list[Neuron]:
        """Get Layer's Neurons list"""
        return self.neurons

    def setNeuronList(self, new_neurons: list[Neuron]):
        """Set Layer's Neuron with new_neurons"""
        self.neurons = new_neurons

    # getter setter individual neurons
    def getNeuronAtIndex(self, idx: int) -> Neuron:
        """Get Neuron at index idx"""
        return self.neurons[idx]

    def setNeuronAtIndex(self, new_neuron: Neuron, idx: int):
        """Set Neuron at index idx with new_neuron"""
        self.neurons[idx] = new_neuron

    # methods for neurons weight setting

    # set all neuron on neuron list with the same weight
    def setAllNeuronWeights(self, set_weight: list[float]):
        """Set every Neuron's weight the same from the given set_weight"""
        for neuron in self.neurons:
            neuron.setWeight(set_weight)

    def setNeuronsWeights(self, neuron_weight: list[list[float]]):
        """Set every Neuron's weight with neuron_weight at matching index"""
        for i in range(len(self.neurons)):
            self.neurons[i].setWeight(neuron_weight[i])

    # set the weight of individual weight at index
    def setNeuronWeightAtIndex(self, set_weight: list[float], idx: int):
        """Set Neuron's weight at index idx"""
        self.neurons[idx].setWeight(set_weight)

    # calculate given input and return list of values for each neuron
    def calculate(self, input: list[float]) -> list[float]:
        """Calculate layer output from the given input"""
        output = [self.layer_bias] if self.layer_bias else []

        if self.algorithm == "input":
            output.extend(input)
        elif (self.algorithm == "softmax"):
            #softmax returns an array of curvalueinneuron/sum(allvalueinneuron)
            #only use softmax in output layer
            output = softmax(input[1:]) # only take neuron output, not bias
        else:
            for neuron in self.neurons:
                curNeuronCalc = neuron.calculate(input)
                output.append(curNeuronCalc)

        self.input_history = input
        self.output_history = output

        return output

    def calculateErrorTerms(self, nextLayer: Layer=None,
        labels:list[float] = None) -> None:
        """Calculate error terms of each layer
        nextLayer is required if current layer is not output layer

        """
        errors = [0 for _ in self.neurons]
        if self.name.lower() == "output layer":
            # Process output layer error term
            for i in range(len(self.neurons)):
                errors[i] = self.neurons[i].errorTerm(
                    output=self.output_history[i],
                    label=labels[i],
                    output_neuron=True)
        elif self.name.lower() == "input layer":
            # Skip input layer (does not have weights)
            pass
        else:
            # Process hidden layer error term
            weight_offset = 1 # 0th index for bias's weight
            for i in range (len(self.neurons)):
                # Get nth weights of next layer
                nthweights = nextLayer.getNthWeights(weight_offset + i)
                sigma = 0
                # Calculate sigma weight_kh * errorterm_k
                for j in range(len(nextLayer.getNeuronList())):
                    sigma += nthweights[j] \
                        * nextLayer.errorterm_history[j]

                errors[i] = self.neurons[i].errorTerm(
                    output=self.output_history[i + weight_offset],
                    sigma=sigma)
        self.errorterm_history = errors

    def getNthWeights(self, n):
        """Return the nth column of weigths from neurons
        example:          wbias  w1  w2  w3
                           n=0  n=1 n=2 n=3
        Neuron 1 (w's) |[[  0,    2,  3,  4]
        Neuron 2 (w's) | [ -2,    1,  5,  3]
        ...
        Neuron n (w's) | [  1,    1,  1,  2]

        return for Layer.getNthWeights(3) will be
          [4, 3, ..., 2]
        """
        return [neuron.weight[n] for neuron in self.neurons]

    def getErrorTermHistory(self):
        """Return the copy of errorterm history"""
        return self.errorterm_history.copy()

    def update(self, delta_weights: list[list[float]]) -> None:
        """Update each neurons""" 
        #update weights
        if self.name.lower() == "input layer":
            # skip input layer (has no weight)
            return
        for i in range(len(self.neurons)):
            self.neurons[i].update(delta_weights[i])
    
    def calculateNeuronsDeltaWeights(self,
        learning_rate:float) -> list[list[float]]:
        """Return neurons delta weights"""
        delta_weights = [[] for _ in self.neurons]
        # Iterate through all neurons
        if self.name.lower() == "input layer":
            return delta_weights
        for i in range(len(self.neurons)):
            delta_weights[i] = \
            [-1 * learning_rate * self.errorterm_history[i] * 
            inputxji for inputxji in self.input_history]
        return delta_weights

    def __str__(self) -> str:
        """Return class as a string"""
        result = "++++++++++++++++++++++++++++++++++++++++++\n"
        result += f"L-NAME : {self.name}\n"
        result += f"ALGORITHM : {self.algorithm}\n"
        result += f"BIAS : {self.layer_bias}\n"
        result += "++++++++++++++++++++++++++++++++++++++++++\n"
        for neuron in self.neurons:
            result += str(neuron)
        return result

    def __len__(self) -> int:
        """Return how many neuron exist inside layer"""
        return len(self.neurons)
