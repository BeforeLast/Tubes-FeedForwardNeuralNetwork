# input array int weight [1,2,3] sama activation [1,2,3]
# nanti di kaliin dan returnnya 1/1+exp(-hasilkali)
# assume panjang weight dan input pasti sama
import math


def sigmoid(weight, input, bias):
    tot = 0
    for i in range(len(weight)):
        tot += weight[i] * input[i]
    tot += bias
    return 1/(1+math.exp(-tot))