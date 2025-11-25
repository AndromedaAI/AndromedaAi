import numpy as np
import string
from Softmax import my_softmax
from Layernorm import my_layer_norm


def simple_tokenizer(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    vocab = sorted(set(words))
    vocab = ['<pad>', '<unk>'] + vocab
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos, vocab


def positional_encoding(seq_len, dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dim))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads[np.newaxis, ...]


class AndromedaBlock:
    def __init__(self, dim=128, n_heads=4):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads  # ← hier was de fout! . → _

        # GEEN seed meer → echt random!
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
        ff = np.maximum(0, ff)
        ff = ff @ self.ff2
        x = x + ff
        x = my_layer_norm(x)
        return x


class MiniAndromeda:
    def __init__(self, vocab_size, dim=128, n_layers=3):
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = np.random.randn(vocab_size, dim) * 0.02
        self.blocks = [AndromedaBlock(dim=dim) for _ in range(n_layers)]
        self.head = np.random.randn(dim, vocab_size) * 0.02

    def forward(self, idx):
        B, S = idx.shape
        x = self.token_emb[idx] + positional_encoding(S, self.dim)
        mask = np.triu(np.full((S, S), -1e9), k=1)[None, None, :, :]

        for block in self.blocks:
            x = block.forward(x, mask)

        x = my_layer_norm(x)
        return x @ self.head


def generate(model, stoi, itos, start_text, max_new=10):
    words = start_text.lower().split()
    idx = np.array([[stoi.get(w, 1) for w in words]])

    for _ in range(max_new):
        logits = model.forward(idx)[0, -1]
        probs = my_softmax(logits)
        next_id = np.random.choice(len(probs), p=probs)
        words.append(itos.get(next_id, "<unk>"))
        idx = np.append(idx, [[next_id]], axis=1)

    return " ".join(words)

# === TEST ===
text = """
De CPI kwam hoger uit dan verwacht daarom gaat de Fed de rente verhogen.
De NFP was zwak dus de markt verwacht een pauze.
Bij sterke payrolls stijgt de dollar.
Ik hou van grote dikke modellen.
"""

stoi, itos, vocab = simple_tokenizer(text)
model = MiniAndromeda(len(vocab))

print("ANDROMEDA SPREEKT (nog onzin, maar WEL elke keer anders)!")
print(generate(model, stoi, itos, "De CPI kwam hoger uit dan verwacht, daarom gaat de"))
print(generate(model, stoi, itos, "De NFP was"))
print(generate(model, stoi, itos, "Ik hou van grote"))