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
