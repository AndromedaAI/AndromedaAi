import numpy as np


def my_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x



def andromeda_attention(q, k, v, mask=None):

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(k.shape[-1])  # (B, S, S)

    if mask is not None:
        scores = scores + mask

    attn_weights = my_softmax(scores, axis=-1)
    output = attn_weights @ v  # (B, S, dim)
    return output, attn_weights
