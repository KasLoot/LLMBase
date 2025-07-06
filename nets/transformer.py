import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import numpy as np




class Embedder(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.embedding_layer = nn.Embedding(CONFIG["vocab_size"], CONFIG["emb_dim"])
        self.dropout = nn.Dropout(CONFIG["dropout_rate"])

    def forward(self, x):
        return self.dropout(self.embedding_layer(x))
    

def RotaryPositionEmbedding(CONFIG, x=None):
    emb_dim = CONFIG["emb_dim"]
    seq_len = CONFIG["max_sequence"]
    positions = torch.arange(seq_len).unsqueeze(0).transpose(-1, 0)
    print(positions.shape)
    theta = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000)/emb_dim))
    angles = positions * theta
    print(positions.shape, theta.shape, angles.shape)
    return theta

def get_rotary_matrix(seq_len, hidden_dim):
    """
    Generates the rotary positional embedding matrix.

    Args:
        seq_len (int): The length of the sequence.
        hidden_dim (int): The dimensionality of the hidden state.

    Returns:
        np.ndarray: The rotary matrix of shape (seq_len, hidden_dim, hidden_dim).
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, hidden_dim, 2) * -(np.log(10000.0) / hidden_dim))
    angles = position * div_term
    print(position.shape, div_term.shape, angles.shape)




CONFIG = {
    "batch_size": 1,
    "max_sequence": 512,
    "emb_dim": 8,  
    "num_head": 8,
    "dropout_rate": 0.1,
    "num_block": 6,
    "input_mask": None,
    "output_mask": None,
    "vocab_size": 200019,
}


RotaryPositionEmbedding(CONFIG)
get_rotary_matrix(seq_len=CONFIG["max_sequence"], hidden_dim=CONFIG["emb_dim"])

# div_term = torch.exp(torch.arange(0, 8, 2) * (-math.log(10000.0) / 8))
# print(div_term)