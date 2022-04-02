from random import sample
from typing import Any
from algorithm.Error import SSE, CrossEntropy
from algorithm.Util import arrayAdd, progressBar
from classes.Layer import Layer
import json
import os
import graphviz
from pdf2image import convert_from_path
from IPython.display import display, Image
import numpy as np
import sys

class FFNN:
    """Feed Forward Neural Network: class for Feed Forward
    Neural Network Model"""
    # Properties
    name: str = None
    layers: list[Layer] = None
    learning_rate: float = None
    dot: graphviz.Digraph = None

    # Methods
    def __init__(self, file_path: str = None) -> None:
        """Class constructor"""
        if file_path is None:
            self.layers = []
            self.name = ""
            self.learning_rate = None
        else:
            try:
                self.load(file_path)
                print(f"NOTICE: MODEL {self.name} LOADED SUCCESSFULLY!")
            except FileNotFoundError as fnfe:
                print("NOTICE: UNABLE TO LOAD MODEL!")
                print(fnfe)

    def addLayer(self, layer: Layer) -> None:
        """Add layer to model"""
        self.layers.append(layer)

    def train(self,
        data: list[list[float]], label: list[list[float]],
        epoch: int = 1, batch_size: int = None,
        treshold: float = None, verbose: bool = True,) -> None:
        """Train model with given input
        data        : list of data
        label       : list of label
        epoch       : maximum iteration
        batch_size  : how many data to be tested before updating weight
        treshold    : maximum value of cumulated error
        """
        if batch_size is None:
            batch_size = len(data)
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be an integer larger than 0")
        if batch_size > len(data):
            raise ValueError(
                "Batch size cannot be larger than training data size")
        for repeat in range(epoch):
            if verbose:
                print(f"\nEpoch {repeat} / {epoch}")
            # Get sample data
            random_data_idx = sample(range(len(data)), batch_size)
            # Reset cumulative error
            sigma_error = 0
            # Reset sigma deltaw
            sigma_deltaw = [
                [[0.0 for weight in neuron.getWeight()]
                    for neuron in layer.getNeuronList()]
                        for layer in self.layers]
            for nthbatch in range(batch_size):
                # Forward Propagation
                ## Do prediction
                output = self.predict(data[random_data_idx[nthbatch]])
                train_label = label[random_data_idx[nthbatch]]
                ## Add error to cumulative error
                if self.layers[-1].getAlgorithm().lower() == "softmax":
                    # Use Cross Entropy for softmax
                    sigma_error += CrossEntropy(train_label, output)
                else:
                    # Use SSE for other algorithm
                    sigma_error += SSE(train_label, output)
                batch_deltaw = []
                # Back Propagation
                # Output errorterm calculation
                self.layers[-1].calculateErrorTerms(labels=train_label)
                # Output weight calculation
                batch_deltaw = batch_deltaw + [self.layers[-1]\
                        .calculateNeuronsDeltaWeights(self.learning_rate)]
                
                # Hidden calculation
                for layer_idx in reversed(range(len(self.layers)-1)):
                    # Iterate through (n-1)th layer to 0th layer
                    # Calculate error term
                    self.layers[layer_idx].calculateErrorTerms(
                        nextLayer=self.layers[layer_idx+1])
                    # Calculate layer delta weight
                    batch_deltaw = [self.layers[layer_idx]\
                        .calculateNeuronsDeltaWeights(self.learning_rate)]\
                        + batch_deltaw
                
                # Add batch_deltaw to sigma_deltaw
                sigma_deltaw = arrayAdd(sigma_deltaw, batch_deltaw)
                if verbose:
                    to_write = ""
                    to_write += f" loss:{round(sigma_error, 5)}"
                    sys.stdout.write(f"\r{nthbatch+1}/{batch_size} {progressBar(nthbatch+1, batch_size)}{to_write}")
                    sys.stdout.flush()

            self.update(sigma_deltaw)
            
            if (treshold and sigma_error <= treshold):
                # If SSE <= error treshold, break from training
                print()
                print(f'Stopped training after {repeat+1} epoch')
                print(f'Current cumulative error: {sigma_error}')
                return
        print()
        print(f'Model successfully trained!')
        print(f'Current cumulative error: {sigma_error}')

    def predict(self, data: list[float]) -> list[float]:
        """Predict output from given input
        example:
          input = [3.14, 1.618, 2.718]

        """
        result = data
        i = 1
        for layer in self.layers:
            result = layer.calculate(result)
            i += 1
        return result

    def batch_predict(self, batch: list[list[float]]) -> list[list[float]]:
        """Predict outputs from given inputs
        example:
          input = [[1, 2, 3], [4, 5, 6], [3.14, 1.618, 2.718]]
        """
        results = []
        for data in batch:
            results.append(self.predict(data))
        return results

    def visualize(self, filename: str = "Model") -> Any:
        """ Visualize FFNN Model"""
        self.dot = graphviz.Digraph(
            self.name,
            comment=self.name,
            graph_attr={'rankdir': 'LR'},
        )
        self.dot.filename = filename if filename.endswith("gv") \
            else filename + ".gv"
        neuron_names = []  # place to store list of all neuron names

        # Make Nodes
        for layer in self.layers:
            neurons = []  # list to store neuron name per layer
            if(layer.name == "input"):
                for i in range(len(layer)):
                    neuron_name = layer.name + "-" + str(i + 1)
                    self.dot.node(neuron_name, neuron_name)
                    neurons.append(neuron_name)
            else:
                for idx, neuron in enumerate(layer.getNeuronList()):
                    neuron_name = layer.name + "-" + str(idx + 1)
                    neuron_item = neuron_name + " : " + \
                        layer.algorithm + "=" + \
                        str(neuron.weight)
                    self.dot.node(neuron_name, neuron_item)
                    neurons.append(neuron_name)
            neuron_names.append(neurons)

        # Make Edges
        for idx in range(len(neuron_names)):
            if(idx > 0):
                for i in (neuron_names[idx]):
                    for j in (neuron_names[idx-1]):
                        self.dot.edge(j, i)

        # Remove existing visualization
        SAVE_DIRECTORY = ".\\file\\visualization\\"
        if os.path.exists(f'{SAVE_DIRECTORY}{filename}'):
            os.remove(f'{SAVE_DIRECTORY}{filename}')
        if os.path.exists(f'{SAVE_DIRECTORY}{filename}.pdf'):
            os.remove(f'{SAVE_DIRECTORY}{filename}.pdf')

        # Saving Graph
        self.dot.render(directory=SAVE_DIRECTORY, view=False)

        # Create PNG from PDF and delete PDF and GV file
        POPPLER_PATH = '.\\ext_lib\\poppler\\Library\\bin'

        # Saving Image
        images = convert_from_path(
            f'{SAVE_DIRECTORY}{filename}.gv.pdf',
            1000,
            poppler_path=POPPLER_PATH
        )

        for i, image in enumerate(images):    
            image.save(f"{SAVE_DIRECTORY}{filename}.png", "PNG")

        # Deleting PDF and GV
        os.remove(f'{SAVE_DIRECTORY}{filename}.gv')
        os.remove(f'{SAVE_DIRECTORY}{filename}.gv.pdf')

        display(Image(
            f'{SAVE_DIRECTORY}{filename}.png',
            width=1000, height=1000)
        )
    
    def update(self, delta_weights:list[list[list[float]]]):
        """Update neurons's weights"""
        for i in range(len(self.layers)):
            self.layers[i].update(delta_weights[i])
    
    def setLearningRate(self, learning_rate:float):
        """Set model's learning rate"""
        self.learning_rate = learning_rate

    def load(self, file_path: str = "model.json", ) -> None:
        """Load model from external file"""
        if (os.path.exists(file_path)):
            with open(file_path, 'r') as file:
                data = file.read()
                data = json.loads(data)
                self.layers = []

                # Create Model
                self.name = data['name']

                # Create input layer
                self.addLayer(
                    Layer("input",
                          data['inputLayerNeuron'],
                          False,
                          "Input Layer",
                          data['inputLayerBias'],)
                )

                # Create hidden layer
                for layer_data in data['hiddenLayers']:
                    hLayer = Layer(
                        layer_data['algorithm'],
                        layer_data['neurons'],
                        False,
                        'Hidden Layer-' + str(layer_data['id']),
                        layer_data['bias'],
                    )
                    hLayer.setNeuronsWeights(layer_data['weights'])
                    self.addLayer(hLayer)

                # Create output layer
                oLayer = Layer(
                    data['outputLayers']['algorithm'],
                    data['outputLayers']['neurons'],
                    False,
                    'Output Layer',
                    None,
                )
                oLayer.setNeuronsWeights(data['outputLayers']['weights'])
                self.addLayer(oLayer)

                file.close()
        else:
            raise FileNotFoundError("Invalid path")

    def save(self, model_name: str="TrainModel", file_path: str = "model.json", ) -> None:
        """Save model to external file"""
        #dictionary that will be written to .json file
        model_temp = {}

        model_temp["name"] = model_name

        #get all layers
        all_layers = self.layers

        #get input layer 
        input_layer = all_layers[0]
        
        #get number of neuron in input layer
        input_layer_length = len(input_layer.getNeuronList())
        model_temp["inputLayerNeuron"] = input_layer_length

        #get input layer bias 
        input_layer_bias = input_layer.layer_bias
        model_temp["inputLayerBias"] = input_layer_bias

        #list of hiddenlayers
        hidden_layers = []

        #loop over the model layers 
        for i in range(1, len(all_layers) - 1) : 
            #one hidden layer 
            hidden_layer = {}

            #current_layer
            current_layer = all_layers[i]

            #get list of neurons
            neurons = current_layer.getNeuronList()

            #assigning hidden_layer attribute 
            hidden_layer["id"] = i
            hidden_layer["neurons"] = len(neurons)
            hidden_layer["algorithm"] = len(current_layer.getAlgorithm())
            hidden_layer["bias"] = current_layer.layer_bias
            
            #list of weights
            weights = []
            #loop over neurons 
            for neuron in neurons : 
                weights.append(neuron.weight)
            
            hidden_layer["weights"] = weights

            hidden_layers.append(hidden_layer)

        model_temp["hiddenLayers"] = hidden_layers

        #initiate output_layer that will be written
        output = {}

        #get output layer
        output_layer = all_layers[len(all_layers) - 1]

        output_layer_neurons = output_layer.getNeuronList() 
        output["neurons"] = len(output_layer_neurons)
        output["algorithm"] = output_layer.algorithm
        
        output_weights = []
        
        for output_neuron in (output_layer_neurons) :
            output_weights.append(output_neuron.weight)

        output["weights"] = output_weights        
        model_temp["outputLayers"] = output
        # Serializing json 
        json_object = json.dumps(model_temp, indent = 2)
        
        # Writing to sample.json
        with open(file_path, "w") as outfile:
            outfile.write(json_object)   

    def __str__(self) -> str:
        """Return class as a string"""
        result = "==========================================\n"
        result += f"MODEL : {self.name}\n"
        result += f"LEARNING_RATE : {self.learning_rate}\n"
        result += "==========================================\n"
        for layer in self.layers:
            result += str(layer)

        return result