import torch
from torch import nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model :int = 512
    max_sql :int = 512
    vocab :int = 201088
    num_heads :int = 8
    d_head :int = 1024
    hidden_layers :int = 4
    hidden_size :int = d_model * 2
    drop_rate :float = 0.1
    transformer_blocks :int = 4
    dist_experts :int = 4
    shared_experts :int = 1
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"



class Embedding(nn.Module):
    def __init__(self, CFG :ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(CFG.vocab, CFG.d_model)
        self.drop = nn.Dropout(CFG.drop_rate)
    

    def forward(self, x :torch.Tensor):
        x = self.drop(self.embedding(x))
        return x
    

class PositionEmbedding(nn.Module):
    def __init__(self, CFG):
        super().__init__()

    
    def forward(self, x):
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, CFG:ModelConfig):
        self.d_model = CFG.d_model
        self.num_heads = CFG.num_heads
        self.d_head = CFG.d_head
        
        self.wqkv = nn.Linear(self.d_model, self.d_model*3)
        self.wo = nn.Linear(self.num_heads * self.d_head, self.d_model)
