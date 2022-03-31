from math import floor


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
def findConfusionMatrix(x,y):
    confusionMatrix = [[0,0,0], [0,0,0], [0,0,0]]
    for i in range (len(x)):
        #intinya, klo x[i] =1 and y[i] = 1, confusion matrix[1][1] ++
        #gitu buat semua elemen
        row = x[i]
        col = y[i]
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

#return tp+fp+tn+fn
def getTotal(confusionMatrix, index):
    return getTruePositive(confusionMatrix, index) + getFalsePositive(confusionMatrix, index) + getTrueNegative(confusionMatrix, index) + getFalseNegative(confusionMatrix, index)

def getAccuracy(confusionMatrix, index):
    #pake index, buat indicate 0,1,2
    # (tp+tn)/(tp+fp+tn+fn)
    if (getTotal(confusionMatrix, index) == 0):
        return 0
    return (getTruePositive(confusionMatrix, index) + getTrueNegative(confusionMatrix, index)) / (getTotal(confusionMatrix, index))

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

def getF1Score(confusionMatrix, index):
    #2* precision*recall/(precision+recall)
    if (getPrecision(confusionMatrix, index) + getRecall(confusionMatrix, index) == 0):
        return 0
    return 2 * (getPrecision(confusionMatrix, index) * getRecall(confusionMatrix, index)) / (getPrecision(confusionMatrix, index) + getRecall(confusionMatrix, index))