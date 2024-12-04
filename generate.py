# generate.py

from model import BigramNeuralNetwork
import torch
import torch.nn.functional as F
from config import block_size

def generate_text(model, start_text, max_new_tokens):
    model.eval()
    idx = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        logits
