import math
# array of output
# output: array of errors
def softmaxError(o):
    err = []
    for i in o:
        err.append(-math.log(i))
    return err

# array of target
# array of output
# both have same length
# output: error value
def SSE(t:list[float], o: list[float]):
    """Sum Squared Error:
    E = 1/2 * sigma(0->k, (tk - to)**2)"""
    err = 0
    for i in range(len(t)):
        err += t[i] - o[i]
    err *= 0.5
    return err

def derSSE(target_j:float, output_j:float):
    return output_j - target_j