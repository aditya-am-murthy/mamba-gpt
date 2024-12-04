# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from config import n_embed, n_heads, dropout

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.keys = nn.Linear(n_embed, head_size)
        self.queries = nn.Linear(n_embed, head_size)
        self.values = nn.Linear(n_embed, head_size)
        self.head_size = head_size
        self.n_embed = n_embed
        self.register_buffer(
            "tril", torch.tril(torch.ones((block_size, block_size))).to(device)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)
        wei = k @ q.transpose(-1, -2) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.log(torch.exp(wei) + 1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.head_size = n_embed // n_heads
        self.sa_head = Mamba(
            d_model=n_embed,
            d_state=16,
            d_conv=4,
            expand=2
        ).to(device)
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class BigramNeuralNetwork(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads=n_heads) for _ in range(n_layers)]
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
