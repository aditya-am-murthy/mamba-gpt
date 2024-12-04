# config.py

import torch
import nltk
from nltk.corpus import brown

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
epochs = 1
lr = 1e-3
batch_size = 48
block_size = 2048
stride = block_size // 2  # Example stride
print_iters = 100
eval_iters = 10
n_embed = 384
n_heads = 6
n_layers = 6
dropout = 0.2
bos_token = "<BOS>"
eos_token = "<EOS>"

# Brown Corpus Data Processing
nltk.download("brown")
chars = sorted(list(set(" ".join(brown.words(fileid)) for fileid in brown.fileids()) + bos_token + eos_token))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
stoi[bos_token] = len(chars) - 2  # Assign unique index for BOS
stoi[eos_token] = len(chars) - 1  # Assign unique index for EOS
itos[len(chars) - 2] = bos_token
itos[len(chars) - 1] = eos_token