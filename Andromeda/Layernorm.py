import numpy as np

def my_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def my_layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

class AndromedaBlock:
    def __init__(self, dim=64, n_heads=4):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads


        self.Wq = np.random.randn(dim, dim) * 0.02
        self.Wk = np.random.randn(dim, dim) * 0.02
        self.Wv = np.random.randn(dim, dim) * 0.02
        self.Wo = np.random.randn(dim, dim) * 0.02

        self.ff1 = np.random.randn(dim, dim*4) * 0.02
        self.ff2 = np.random.randn(dim*4, dim) * 0.02

    def attention(self, x, mask=None):
        B, S, D = x.shape


        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv


        q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)


        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)


        if mask is not None:
            scores = scores + mask

        weights = my_softmax(scores, axis=-1)
        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        return out @ self.Wo

    def forward(self, x, mask=None):

        x = x + self.attention(x, mask)
        x = my_layer_norm(x)


        ff = x @ self.ff1
        ff = np.maximum(0, ff)      # ReLU
        ff = ff @ self.ff2
        x = x + ff
        x = my_layer_norm(x)
        return x
