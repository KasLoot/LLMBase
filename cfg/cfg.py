import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tiktoken

MASTER_CONFIG = {
    "vocab_size": tiktoken.get_encoding("cl100k_base").n_vocab,    # Size of the vocabulary
    'batch_size': 8,          # Number of batches to be processed at each random split
    'context_window': 32,
    'd_model': 128,
    'n_heads': 8,
    'epochs': 10,          # Number of training epochs
    'log_interval': 10,      # Log information every 10 batches during training
    'batch_size': 32
}
