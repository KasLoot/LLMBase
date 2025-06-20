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
        x = self.linear1(x.transpose(1, 2)).transpose(1, 2)
        out = self.swiglu(x)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.rms_norm1(out)
        out = x + out
        out = self.rms_norm2(out)
        out = self.linear3(out)
        return out
    

class Up_UNet(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.upsample = nn.Linear(CONFIG["max_sequence"], CONFIG["out_max_sequence"])
        self.linear1 = nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"])
        self.linear2 = nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"])
        self.linear3 = nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"])
        self.linear4 = nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"])
        self.linear5 = nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"])
        self.linear6 = nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"])
        self.linear7 = nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"])
        self.linear8 = nn.Linear(CONFIG["emb_dim"], CONFIG["emb_dim"])
        
        # self.dropout = nn.Dropout(CONFIG["dropout_rate"])
        self.swiglu = SwiGLU(CONFIG["emb_dim"])
        self.rms_norm1 = nn.RMSNorm(CONFIG["emb_dim"])
        self.u_rms_norm1 = nn.RMSNorm(CONFIG["emb_dim"])
        self.u_rms_norm2 = nn.RMSNorm(CONFIG["emb_dim"])
        self.u_rms_norm3 = nn.RMSNorm(CONFIG["emb_dim"])
        self.u_rms_norm4 = nn.RMSNorm(CONFIG["emb_dim"])
        self.u_rms_norm5 = nn.RMSNorm(CONFIG["emb_dim"])
        self.u_rms_norm6 = nn.RMSNorm(CONFIG["emb_dim"])
        self.u_rms_norm7 = nn.RMSNorm(CONFIG["emb_dim"])
        self.u_rms_norm8 = nn.RMSNorm(CONFIG["emb_dim"])
        self.final = nn.Linear(CONFIG["emb_dim"], CONFIG["vocab_size"])

    def forward(self, x: torch.Tensor):
        x = self.upsample(x.transpose(1, 2)).transpose(1, 2)
        x = self.swiglu(x)
        x = self.rms_norm1(x)
        # U-net begin
        l1 = self.linear1(x)
        out = self.u_rms_norm1(l1)
        l2 = self.linear2(out)
        out = self.u_rms_norm2(l2)
        l3 = self.linear3(out)
        out = self.u_rms_norm3(l3)
        out = self.linear4(out)
        out = self.u_rms_norm4(out)
        out = self.linear5(out)
        out = self.u_rms_norm5(l3 + out)
        out = self.linear6(out)
        out = self.u_rms_norm6(l2 + out)
        out = self.linear7(out)
        out = self.u_rms_norm7(l1 + out)
        out = self.linear8(out)
        out = self.u_rms_norm8(out)
        out = self.final(out)
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
        self.up_ffn = Up_UNet(CONFIG)

    
    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.rms_norm1(x)
        for block in self.llama3_blocks:
            x = block(x)
        x = self.up_ffn(x)
        return x
