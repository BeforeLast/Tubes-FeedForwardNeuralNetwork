from random import sample
from typing import Any
from algorithm.Error import SSE
from algorithm.Util import arrayAdd
from classes.Layer import Layer
import json
import os
import graphviz
from pdf2image import convert_from_path
from IPython.display import display, Image
import numpy as np

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
            self.learning_rate = 0.0
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
        epoch: int = 1, batch_size: int = 1,
        treshold: float = None) -> None:
        """Train model with given input"""
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be an integer larger than 0")
        if batch_size > len(data):
            raise ValueError(
                "Batch size cannot be larger than training data size")
        for repeat in range(epoch):
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
                sigma_error += SSE(train_label,output)
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
            self.update(sigma_deltaw)
            
            if (treshold and SSE(train_label, output) <= treshold):
                # If SSE <= error treshold, break from training
                print(f'Model successfully trained in {repeat} epoch')
                print(f'Current cumulative error: {SSE(train_label, output)}')
                return
        print(f'Model successfully trained!')

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

    def save(self, file_path: str = "model.json", ) -> None:
        """Save model to external file"""
        pass

    def __str__(self) -> str:
        """Return class as a string"""
        result = "==========================================\n"
        result += f"MODEL : {self.name}\n"
        result += f"LEARNING_RATE : {self.learning_rate}\n"
        result += "==========================================\n"
        for layer in self.layers:
            result += str(layer)

        return result