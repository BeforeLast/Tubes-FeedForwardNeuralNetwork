from algorithm.Error import error, softmaxError
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
        for i in range(n_neurons):
            self.neurons.append(Neuron(algorithm, []))

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
            output = softmax(input)
        else:
            for neuron in self.neurons:
                curNeuronCalc = neuron.calculate(input)
                output.append(curNeuronCalc)

        return output

    def update(self, outputArr, targetArr, learn_rate) -> None:
        """Update each neurons"""
        #find neuron outputs
        neuron_outputs = []
        for neuron in self.neurons:
            neuron_outputs.append(neuron.calculate(targetArr))

        # outputArr is the output of current layer
        # targetArr is array of actual target / target dari next layer
        if self.name.lower() == "output layer":
            if self.algorithm == "softmax":
                error_array = softmaxError(outputArr)
            else:
                errTerm = error(targetArr, outputArr)
                error_array = [errTerm for i in range (len(outputArr))]
        else: #hidden layer
            # get neuron errors
            error_array = []
            for i in range (len(self.neurons)):
                error_array.append(self.neurons[i].getHiddenError(outputArr, neuron_outputs[i]))
        
        #update weights
        for i in range(len(self.neurons)):
            self.neurons[i].update(error_array, neuron_outputs[i], learn_rate)

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
