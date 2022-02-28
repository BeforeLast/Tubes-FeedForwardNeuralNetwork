# input array int weight [1,2,3] sama activation [1,2,3]
# nanti di kaliin dan returnnya yg paling gede dari 0 sm hasil kali
# assume panjang weight dan input pasti sama
def relu(weight, input):
    tot = 0
    for i in range(len(weight)):
        tot += weight[i] * input[i]
    return max(0, tot)
