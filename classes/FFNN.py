from typing import Any
from classes.Layer import Layer
import json
import os


class FFNN:
    """Feed Forward Neural Network: class for Feed Forward
    Neural Network Model"""
    # Properties
    name: str = None
    layers: list = None
    learning_rate: float = None

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
        result = input
        i = 1
        for layer in self.layers:
            result = layer.calculate(result)
            i += 1
        return result

    def visualize(self) -> Any:
        """Visualize FFNN model"""
        pass

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
