import nntplib
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
        #create a layer with n neurons, each having [] as initial_weight
        self.alogrithm = algorithm
        self.trainable = trainable
        self.name = name
        self.layer_bias = layer_bias
        for i in range (n_neurons):
            self.neurons.append(Neuron(algorithm,[],layer_bias))
    
    def calculate(self, input:list[float]) -> list[float]:
        """Calculate layer output from the given input"""
        return []
    
    def update(self,) -> None:
        """Update each neurons"""
        pass
    
