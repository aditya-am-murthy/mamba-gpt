# train.py

import torch
from tqdm import tqdm
from model import BigramNeuralNetwork
from data_preprocessing import prepare_data, split_data, get_batch
from config import lr, epochs, batch_size, device

def train(model, optimizer, train_data, val_data):
    max_iters = len(train_data) // batch_size * epochs
    losses_data = {"train": [], "test": []}
    
    for iter in tqdm(range(max_iters)):
        if iter % eval_iters == 0:
            losses = estimate_loss(model, train_data, val_data)
            losses_data["train"].append(losses["train"])
            losses_data["test"].append(losses["test"])

        if iter % print_iters == 0:
            losses = estimate_loss(model, train_data, val_data)
            torch.save(model.state_dict(), f"./model_checkpoint_{iter}.pt")
            losses_data["train"].append(losses["train"])
            losses_data["test"].append(losses["test"])

        # Get training batch
        xb, yb = get_batch("train", train_data, val_data, batch_size)

        # Calculate loss and update the model
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

def estimate_loss(model, train_data, val_data):
    model.eval()
    losses = {"train": 0, "test": 0}
    for split in ["train", "test"]:
        loss = 0
        for _ in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, batch_size)
            _, batch_loss = model(X, Y)
            loss += batch_loss.item()
        losses[split] = loss / eval_iters
    model.train()
    return losses
