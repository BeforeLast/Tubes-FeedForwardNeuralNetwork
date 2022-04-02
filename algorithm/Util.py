from calendar import c
from math import floor
import json


def arrayAdd(A, B):
    """Recursively add array with matching dimension"""
    if type(A) != type([]):
        return A+B
    else:
        if len(A) != len(B):
            raise IndexError("Mismatch array size")
        return [arrayAdd(A[i],B[i]) for i in range(len(A))]

def progressBar(value, max_value):
    """Return the progress bar of the given value and max_value"""
    value_percent = value/max_value
    bar_length = 30
    value_char = floor(bar_length * value_percent)
    if value_char == bar_length:
        remain_char = ''
    else:
        remain_char = '>' + '.' * (bar_length - value_char - 1)
    return f"[{'='*value_char}{remain_char}]"

'''
find confusion matrix between int array x and int array y
make sure x and y has same length
return 3x3 matrix karena ad 3 kelas
intinya gini
setosa, versicolor, virginica : 0,1,2 <- in that order, i checked
the confusion matrix would be kyk gini
            setosa  versicolor  virginica
setosa      0       0           0
versicolor  0       0           0
virginica   0       0           0
referensi gw: https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/#:~:text=The%20confusion%20matrix%20is%20a,and%20False%20Negative(FN).
'''
#x TRUE, y PREDICTION
def findConfusionMatrix(x,y):
    confusionMatrix = [[0,0,0], [0,0,0], [0,0,0]]
    for i in range (len(x)):
        #intinya, klo x[i] =1 and y[i] = 1, confusion matrix[1][1] ++
        #gitu buat semua elemen
        row = int(round(x[i]))
        col = int(round(y[i]))
        confusionMatrix[row][col] += 1
    return confusionMatrix

#get true positive for a specific index
#confusion matrix harusnya 3x3, indexnya: 0,1,2
def getTruePositive(confusionMatrix, index):
    return confusionMatrix[index][index]

#note: confMatrix 3x3
def getFalseNegative(confusionMatrix, index):
    if index == 0:
        return confusionMatrix[0][1] + confusionMatrix[0][2]
    elif index == 1:
        return confusionMatrix[1][0] + confusionMatrix[1][2]
    #else
    return confusionMatrix[2][0] + confusionMatrix[2][1]

def getFalsePositive(confusionMatrix, index):
    if index == 0:
        return confusionMatrix[1][0] + confusionMatrix[2][0]
    elif index == 1:
        return confusionMatrix[0][1] + confusionMatrix[2][1]
    #else
    return confusionMatrix[0][2] + confusionMatrix[1][2]

def getTrueNegative(confusionMatrix, index):
    if index == 0:
        return confusionMatrix[1][1] + confusionMatrix[1][2] + confusionMatrix[2][1] + confusionMatrix[2][2]
    elif index == 1:
        return confusionMatrix[0][1] + confusionMatrix[0][2] + confusionMatrix[2][0] + confusionMatrix[2][2]
    #else
    return confusionMatrix[0][0] + confusionMatrix[0][1] + confusionMatrix[1][0] + confusionMatrix[1][1]

def getTotal(confusionMatrix):
    sum = 0
    for row in confusionMatrix:
        for element in row:
            sum+= element
    return sum

def getAccuracy(confusionMatrix):
    # sum of diagonal / total
    # gw dpt dari sini: https://arxiv.org/pdf/2008.05756.pdf#:~:text=Accuracy%20is%20one%20of%20the,computed%20from%20the%20confusion%20matrix.&text=The%20formula%20of%20the%20Accuracy,confusion%20matrix%20at%20the%20denominator.
    return ((confusionMatrix[0][0] + confusionMatrix[1][1] + confusionMatrix[2][2]) / getTotal(confusionMatrix))

def getPrecision(confusionMatrix, index):
    #tp + tp/fp
    if (getTruePositive(confusionMatrix, index) + getFalsePositive(confusionMatrix, index) == 0):
        return 0
    return (getTruePositive(confusionMatrix, index)) / (getTruePositive(confusionMatrix, index) + getFalsePositive(confusionMatrix, index))

def getRecall(confusionMatrix, index):
    #tp /(tp+fn)
    if (getTruePositive(confusionMatrix, index) + getFalseNegative(confusionMatrix, index) == 0):
        return 0
    return (getTruePositive(confusionMatrix, index)) / (getTruePositive(confusionMatrix, index) + getFalseNegative(confusionMatrix, index))

def getMacroPrecision(confusionMatrix):
    sum = 0
    #matrixnya always nxn, in this case 3x3 so this is fine
    for i in range(len(confusionMatrix)):
        sum += getPrecision(confusionMatrix,i)
    return sum/len(confusionMatrix)

def getMacroRecall(confusionMatrix):
    sum = 0
    #matrixnya always nxn, in this case 3x3 so this is fine
    for i in range(len(confusionMatrix)):
        sum += getRecall(confusionMatrix,i)
    return sum/len(confusionMatrix)

def getMacroF1Score(confusionMatrix):
    #2* precision*recall/(precision+recall)
    if (getMacroPrecision(confusionMatrix) + getMacroRecall(confusionMatrix)) == 0:
        return 0
    return 2 * (getMacroPrecision(confusionMatrix) * getMacroRecall(confusionMatrix)) / (getMacroPrecision(confusionMatrix) + getMacroRecall(confusionMatrix))