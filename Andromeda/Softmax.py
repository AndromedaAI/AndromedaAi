import numpy as np
def my_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    softmax = exp_x / sum_exp_x
    return softmax
