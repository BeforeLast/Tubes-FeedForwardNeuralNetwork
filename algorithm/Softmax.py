# input full matriks hidden layer terakhir (data) dan hasilnya (activation input) 
# nanti di kaliin dan returnnya probabilitas per neuron
# assume ukuran data dan input sama
import math
def softmax(data, activation_input):
    all = []
    for i in range(len(data)):
        # print(i)
        temp = 0
        for j in range(len(data[0])):
            temp += data[i][j]*activation_input[i][j]
        all.append(temp)
    print(all)
    map(lambda x: math.exp(x), all)
    tot = sum(all)
    for i in range(len(all)):
        all[i] = all[i]/tot
    return all

def toExp(a):
    return math.exp(a)