def arrayAdd(A, B):
    if type(A) != type([]):
        return A+B
    else:
        if len(A) != len(B):
            raise IndexError("Mismatch array size")
        return [arrayAdd(A[i],B[i]) for i in range(len(A))]
