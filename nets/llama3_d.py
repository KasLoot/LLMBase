import torch
import torch.nn as nn
import torch.functional as F

from .llama3 import Llama3_Block, SwiGLU
from .net import EmbeddingModel

class Text_Diffusion(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()

    
    def forward(self, x):
        return x
    


class Up_FFN(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.linear1 = nn.Linear(CONFIG["max_sequence"], CONFIG["out_max_sequence"])
        self.linear2 = nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"])
        self.dropout = nn.Dropout(CONFIG["dropout_rate"])
        self.swiglu = SwiGLU(CONFIG["emb_dim"])
        self.rms_norm1 = nn.RMSNorm(CONFIG["emb_dim"])
        self.rms_norm2 = nn.RMSNorm(CONFIG["emb_dim"])
        self.linear3 = nn.Linear(CONFIG["emb_dim"], CONFIG["vocab_size"])

    def forward(self, x: torch.Tensor):
        out = self.linear1(x.transpose(1, 2)).transpose(1, 2)
        out = self.swiglu(x)
        out = self.dropout(x)
        
        out = self.linear2(x)
        out = self.rms_norm1(x)
        out = x + out
        out = self.rms_norm2(out)
        out = self.linear3(out)
        return out


class Text_Diffusion(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        


class Llama3_D(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.embedding = EmbeddingModel(CONFIG["emb_dim"], CONFIG["vocab_size"], CONFIG["dropout_rate"])
        self.llama3_blocks = nn.ModuleList([Llama3_Block(CONFIG) for _ in range(CONFIG["num_block"])])
        self.rms_norm1 = nn.RMSNorm(CONFIG["emb_dim"])
        self.up_ffn = Up_FFN(CONFIG)

    
    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.rms_norm1(x)
        for block in self.llama3_blocks:
            x = block(x)
        x = self.up_ffn(x)
        return x
