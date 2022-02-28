# input full matriks hidden layer terakhir (data) dan
# hasilnya (activation input) 
# nanti di kaliin dan returnnya probabilitas per neuron
# assume ukuran data dan input sama
import math


def softmax(outputs):
    map(lambda x: math.exp(x), outputs)
    tot = sum(outputs)
    for i in range(len(outputs)):
        outputs[i] = outputs[i]/tot
    return outputs