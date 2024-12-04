# data_preprocessing.py

import torch
from sklearn.model_selection import train_test_split
from config import block_size, stride, bos_token, eos_token, stoi, itos
import numpy as np
import nltk
from nltk.corpus import brown

def encode(text):
    """Encodes text into token indices."""
    return [stoi[x] for x in text]

def decode(indices):
    """Decodes token indices back to text."""
    return "".join([itos[x] for x in indices])

def prepare_data():
    """Prepare data from the Brown corpus."""
    brown_text = "".join([bos_token + " ".join(brown.words(fileid)) + eos_token for fileid in brown.fileids()])
    data = torch.tensor(encode(brown_text), dtype=torch.long)
    return data

def chunk_data_with_stride(data, block_size, stride):
    """Create chunks using strides for overlapping sequences."""
    return [data[i:i + block_size] for i in range(0, len(data) - block_size, stride)]

def get_batch(split, train_data, val_data, batch_size):
    """Generate batches of data."""
    data = train_data if split == "train" else val_data
    index = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[ind:ind + block_size] for ind in index])
    y = torch.stack([data[ind + 1:ind + block_size + 1] for ind in index])
    return x, y

def split_data(data, block_size, stride):
    """Split data into training and validation sets."""
    strided_sequences = chunk_data_with_stride(data, block_size, stride)
    train_sequences, val_sequences = train_test_split(strided_sequences, train_size=0.9)
    train_data = torch.cat(train_sequences, dim=0)
    val_data = torch.cat(val_sequences, dim=0)
    return train_data, val_data
