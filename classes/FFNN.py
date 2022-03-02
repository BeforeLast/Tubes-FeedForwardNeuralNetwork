from typing import Any
from classes.Layer import Layer
import json
import os
import graphviz


class FFNN:
    """Feed Forward Neural Network: class for Feed Forward
    Neural Network Model"""
    # Properties
    name: str = None
    layers: list = None
    learning_rate: float = None
    input: list = None
    dot = graphviz.Digraph(name, comment=name,graph_attr={'rankdir':'LR'})

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
                print("NOTICE: MODEL LOADED SUCCESSFULLY!")
            except FileNotFoundError as fnfe:
                print("NOTICE: UNABLE TO LOAD MODEL!")
                print(fnfe)

    def addLayer(self, layer: Layer) -> None:
        """Add layer to model"""
        self.layers.append(layer)

    def train(self, input: list[float], epoch: int = 10, ) -> None:
        """Train model with given input"""
        pass

    def predict(self, input: list[float]) -> list[float]:
        """Predict output from given input
        example:
          input = [3.14, 1.618, 2.718]

        """
        self.input = input
        result = input
        i = 1
        for layer in self.layers:
            result = layer.calculate(result)
            i += 1
        return result

    def visualize(self) -> Any:
        """Visualize FFNN model"""
        neuron_names = [] #place to store list of all neuron names

        """Make Nodes"""
        for layer in self.layers :
            neurons = [] #list to store neuron name per layer
            if(layer.name == "input") :
                for i in range(len(self.input)) :
                    neuron_name = layer.name + "-" + str(i + 1)
                    self.dot.node(neuron_name, neuron_name+" : "+str(self.input[i]))
                    neurons.append(neuron_name)
            else : 
                for idx, neuron in enumerate(layer.getNeuronList()) :
                    neuron_name = layer.name + "-" + str(idx + 1)
                    neuron_item = neuron_name+" : "+layer.algorithm + "=" + str(neuron.weight)
                    self.dot.node(neuron_name, neuron_item)
                    neurons.append(neuron_name)
            neuron_names.append(neurons)

        """Result"""
        neuron_names.append(["result"])
        self.dot.node("result", "Result = " + str(self.predict(self.input)))

        """Make Edges"""
        for idx in range(len(neuron_names)) : 
            if(idx > 0) : 
                for i in (neuron_names[idx]) : 
                    for j in (neuron_names[idx-1]) : 
                        self.dot.edge(j,i)

        print(self.dot.source)
        self.dot.render(directory="file", view=True)

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
                          "input",
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
        """Convert object to string"""
        result = "==========================================\n"
        result += f"MODEL : {self.name}\n"
        result += f"LEARNING_RATE : {self.learning_rate}\n"
        result += "==========================================\n"
        for layer in self.layers:
            result += str(layer)

        return result
