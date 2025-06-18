import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import EmbeddingModel, Positional_Embedding, MultiHeadAttention


class SwiGLU(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear_1 = nn.Linear(emb_dim,emb_dim)
        self.linear_2 = nn.Linear(emb_dim,emb_dim)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)

        return swiglu


class Llama3_FFN(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(0.1)
        self.swiglu = SwiGLU(emb_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Llama3_Block(nn.Module):
    def __init__(self, CONFIG, mask=None):
        super().__init__()
        self.rms_norm1 = nn.RMSNorm(CONFIG["emb_dim"])
        self.position_emb = Positional_Embedding(CONFIG["emb_dim"], CONFIG["max_sequence"], CONFIG["dropout_rate"])
        self.mh_attention = MultiHeadAttention(CONFIG["emb_dim"], CONFIG["num_head"], mask)
        self.rms_norm2 = nn.RMSNorm(CONFIG["emb_dim"])
        self.ffn = Llama3_FFN(CONFIG["emb_dim"])

    def forward(self, x):
        normed = self.rms_norm1(x)
        position_emb = self.position_emb(normed)
        attention_output = self.mh_attention(position_emb, position_emb, x)
        x = x + attention_output
        normed = self.rms_norm2(x)
        ffn_output = self.ffn(normed)
        x = x + ffn_output
        return x
    

class Llama3(nn.Module):
    def __init__(self, CONFIG, mask=None):
        super().__init__()
        self.embedding = EmbeddingModel(CONFIG["emb_dim"], CONFIG["vocab_size"], CONFIG["dropout_rate"])

        self.blocks = nn.ModuleList([Llama3_Block(CONFIG, mask) for _ in range(CONFIG["num_block"])])
        self.rms_norm = nn.RMSNorm(CONFIG["emb_dim"])
        self.linear = nn.Linear(CONFIG["emb_dim"], CONFIG["vocab_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
