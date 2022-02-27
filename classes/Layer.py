import nntplib
from typing_extensions import Self
from classes.Neuron import Neuron


class Layer:
    """Layer Class"""
    # Properties
    alogrithm:str = None
    neurons:list[Neuron] = None
    trainable:bool = None
    name:str = None
    layer_bias:float = None
    # Methods
    def __init__(self, algorithm, n_neurons, trainable, name, layer_bias):
        """Class constructor"""
        #create a layer with n neurons, each having [] as initial_weight (empty array)
        #akan diasumsi weight setting untuk either setiap neuron or individual neuron akan dilakukan di model
        self.alogrithm = algorithm
        self.trainable = trainable
        self.name = name
        self.layer_bias = layer_bias
        for i in range (n_neurons):
            self.neurons.append(Neuron(algorithm,[],layer_bias))
    
    #getter setter neuron list
    def getNeuronList(self) -> list[Neuron]:
        return self.neurons

    def setNeuronList(self, new_neurons: list[Neuron]):
        self.neurons = new_neurons

    #getter setter individual neurons
    def getNeuronAtIndex(self,idx) -> Neuron:
        return self.neurons[idx]

    def setNeuronAtIndex(self, new_neuron:Neuron, idx):
        self.neurons[idx] = new_neuron

    #methods for neurons weight setting

    #set all neuron on neuron list with the same weight
    def setAllNeuronWeights(self, set_weight:list[float]):
        for neuron in self.neurons:
            neuron.setWeight(set_weight)

    #set the weight of individual weight at index
    def setNeuronWeightAtIndex(self, set_weight:list[float],idx):
        self.neurons[idx].setWeight(set_weight)

    #calculate given input and return list of values for each neuron
    def calculate(self, input:list[float]) -> list[float]:
        """Calculate layer output from the given input"""
        output = [float]

        for neuron in self.neurons:
            curNeuronCalc = neuron.calculate(input)
            output.append(curNeuronCalc)

        return output
    
    def update(self,) -> None:
        """Update each neurons"""
        pass
    
