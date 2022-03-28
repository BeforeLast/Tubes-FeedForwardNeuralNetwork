from ast import Mod
from classes.FFNN import FFNN


Model_XOR_Sigmoid = FFNN("./file/models/XOR_Sigmoid.json")
Model_XOR_Sigmoid.setLearningRate(0.01)

res = Model_XOR_Sigmoid.predict([1.0, 0.0])

xor_data = [[0.0,0.0], [0.0,1.0], [1.0, 0.0], [1.0, 1.0]]
label = [[0], [1], [1], [0]]

Model_XOR_Sigmoid.train(xor_data,label, 1000, 1, 0.000000000000001)

res2 = Model_XOR_Sigmoid.predict([1.0, 0.0])

print(res)
print(res2)
