import math

# linear derivative
def derLinear(x):
    return 1

# relu derivative
def derRelu(x):
    return 1 if x >=0 else 0

# sigmoid derivative
def derSigmoid(x):
    return 1/(1+math.exp(-x)) * (1 - 1/(1+math.exp(-x)))

# softmax derivative
def derSoftmax(x, j, targetClass):
    return x if j == targetClass else -1+x