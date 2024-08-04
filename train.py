import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from dataset.dataset import VQADataset
from torch.utils.data import DataLoader
from cfg.cfg import MASTER_CONFIG
from nets.net import LLaMANetwork
import time
import tiktoken

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using GPU')
else:
    device = torch.device("cpu")
    print('Using CPU')


@torch.no_grad()  # Don't compute gradients for this function
def evaluate_loss(model: nn.Module, config=MASTER_CONFIG, val_dataset=None):
    loss_l = []
    # Set the model to evaluation mode
    model.eval()

    for val_example in val_dataset:
        x, y = val_example
        x = x.to(device)
        y = y.to(device)
        # Forward pass
        y_pred = model(x)
        # Compute the loss
        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        loss_l.append(loss.item())

    return np.mean(loss_l)

def train(config=MASTER_CONFIG):
    batch_size = 2
    train_dataset = VQADataset("/home/yuxin/Documents/LLMBase/dataset/task_planning.json")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Define model parameters
    vocab_size = tiktoken.get_encoding('cl100k_base').n_vocab  # Size of the vocabulary
    d_model = 512  # Dimension of the model
    num_layers = 6  # Number of encoder layers
    num_heads = 8  # Number of attention heads
    d_ff = 2048  # Dimension of feed-forward network
    max_len = 5000  # Maximum sequence length
    dropout = 0.1  # Dropout rate
    # Initialize the model
    model = LLaMANetwork(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Initialize the loss function
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    start_time = time.time()

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        i = 0
        for batch in train_dataloader:
            # x, y = train_example[0], train_example[1]
            # x = x.to(device)
            # y = y.to(device)
            # # Forward pass
            # y_pred = model(x)
            # print(x.shape, y.shape, y_pred.shape)
            # break
            # loss = criterion(y_pred, y)
            # # Backward pass
            # loss.backward()
            # # Update the weights
            # optimizer.step()
            # if scheduler is not None:
            #     scheduler.step()
            # i += 1
            # print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}')
        # val_loss = evaluate_loss(model, config)
        # print(f'Epoch {epoch + 1}, Validation Loss {val_loss}')
            break
        break

def main():
    train()

if __name__ == "__main__":
    main()