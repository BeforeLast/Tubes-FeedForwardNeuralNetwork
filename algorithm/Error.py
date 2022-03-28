import math
# array of output
# output: array of errors
def CrossEntropy(label: list[float], prediction: list[float]) -> float:
    """Cross Entropy Error:
    E = sigma(0->k,
          (label_k)*log(prediction_k) + (1 - label_k)*log(1 - prediction_k)
        )
    """
    sigma_err = 0
    for i in range(len(label)):
        if label[i] == 1:
            sigma_err += -math.log(1 - prediction[i])
        else:
            sigma_err += -math.log(prediction[i])
    return sigma_err

def derCrossEntropy(label_j:float, preditiction_j:float) -> float:
    """Return the derivative cross entropy value from the
    given label and prediction
    return prediction_j - 1 if label is 1
    return prediction_j if label is 0
    """
    if label_j == 1:
        return preditiction_j - 1
    else:
        return preditiction_j

# array of target
# array of output
# both have same length
# output: error value
def SSE(label:list[float], prediction: list[float]) -> float:
    """Sum Squared Error:
    E = 1/2 * sigma(0->k, (label_k - prediction_k)**2)"""
    err = 0
    for i in range(len(label)):
        err += (label[i] - prediction[i])**2
    err *= 0.5
    return err

def derSSE(label_j:float, preditiction_j:float) -> float:
    """Return the derivative value of Sum Squared Error:
    return prediction_j - label_j"""
    return label_j - preditiction_j

