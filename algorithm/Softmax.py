# input full matriks hidden layer terakhir (data) dan
# hasilnya (activation input) 
# nanti di kaliin dan returnnya probabilitas per neuron
# assume ukuran data dan input sama
import math


def softmax(inputs: list[float]):
    sigma_ex = sum([math.exp(xi) for xi in inputs])
    outputs = [0 for _ in inputs]
    for i in range(len(inputs)):
        outputs[i] = math.exp(inputs[i])/sigma_ex
    return outputs