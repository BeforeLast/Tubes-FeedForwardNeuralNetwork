# input array int weight [1,2,3] sama activation [1,2,3]
# nanti di kaliin dan returnnya sama kayak nilai hasil kali + bias (int): (1+4+9 + b)
# assume panjang weight dan input pasti sama
def linear(weight, input, bias):
    tot = 0
    for i in range(weight):
        tot += weight[i] * input[i]
    tot += bias
    return tot