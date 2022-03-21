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
def error(t, o):
    err = 0
    for i in range(len(t)):
        err += t[i] - o[i]
    err *= 0.5
    return err