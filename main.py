from classes.FFNN import FFNN

if __name__ == "__main__":
    Model = FFNN('./file/XOR.json')
    print(f'0 XOR 0 = {Model.predict([0,0])}')
    print(f'0 XOR 1 = {Model.predict([0,1])}')
    print(f'1 XOR 0 = {Model.predict([1,0])}')
    print(f'1 XOR 1 = {Model.predict([1,1])}')
    Model.visualize()
