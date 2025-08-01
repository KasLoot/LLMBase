import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import numpy as np

from utils.tokenizer import Tokenizer



class Embedder(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.embedding_layer = nn.Embedding(CFG["vocab_size"], CFG["emb_dim"])
        self.dropout = nn.Dropout(CFG["dropout_rate"])

    def forward(self, x):
        return self.dropout(self.embedding_layer(x))
    





class MultiHeadAttention(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        # self.batch_size = self.CFG["batch_size"]
        # self.seq_len = self.CFG["max_sequence"]
        # self.emb_dim = self.CFG["emb_dim"]
        # self.num_head = self.CFG["num_head"]
        # self.d_r = self.CFG["dropout_rate"]
        self.CFG = CONFIG
        self.d_k = self.CFG["emb_dim"] // self.CFG["num_head"]
        assert self.CFG["emb_dim"] % self.CFG["num_head"] == 0, "embedding dimention should be divisiable by num_head"
        self.Q = nn.Linear(self.CFG["emb_dim"], self.CFG["emb_dim"])
        self.K = nn.Linear(self.CFG["emb_dim"], self.CFG["emb_dim"])
        self.V = nn.Linear(self.CFG["emb_dim"], self.CFG["emb_dim"])
        self.rotary_matrix = self.RotaryPositionEmbedding(self.CFG)


    def RotaryPositionEmbedding(self, CFG):
        emb_dim = CFG["emb_dim"]
        seq_len = CFG["max_sequence"]
        positions = torch.arange(seq_len).unsqueeze(0).transpose(-1, 0)
        # print(positions.shape)
        theta = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000)/emb_dim))
        angles = positions * theta
        # print(positions.shape, theta.shape, angles.shape)

        sine_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        rotary_matrix = torch.zeros((seq_len, emb_dim, emb_dim))

        rotary_matrix[:, torch.arange(0, emb_dim, 2), torch.arange(0, emb_dim, 2)] = cos_angles
        rotary_matrix[:, torch.arange(0, emb_dim, 2), torch.arange(1, emb_dim, 2)] = -sine_angles
        rotary_matrix[:, torch.arange(1, emb_dim, 2), torch.arange(0, emb_dim, 2)] = sine_angles
        rotary_matrix[:, torch.arange(1, emb_dim, 2), torch.arange(1, emb_dim, 2)] = cos_angles

        # Register as buffer so it moves with the model to the correct device
        self.register_buffer('rotary_matrix_buffer', rotary_matrix)
        return rotary_matrix


    def split_heads(self, x:torch.Tensor):
        B, S, D = x.size()
        # Remove assertions that are too rigid - the model should be flexible with batch sizes
        # assert B == self.CFG["batch_size"], "[Attention Module] unmatched batch size"
        # assert S == self.CFG["max_sequence"], "[Attention Module] unmatched sequence length"
        assert D == self.CFG["emb_dim"], "[Attention Module] unmatched embedding dimension"

        return x.view(B, S, self.CFG["num_head"], self.d_k).transpose(1, 2)
    
    def attention_score(self, x_q:torch.Tensor, x_k:torch.Tensor, x_v:torch.Tensor):
        # Get sequence length for indexing rotary matrix
        seq_len = x_q.size(1)
        
        # Apply rotary position embedding using matrix multiplication instead of dot product
        # Use the registered buffer which will be on the correct device
        rotary_matrix = self.rotary_matrix_buffer[:seq_len]  # Get relevant portion
        
        # Apply rotary embedding: (B, S, D) @ (S, D, D) -> (B, S, D)
        x_q_rotary = torch.einsum('bsd,sde->bse', x_q, rotary_matrix)
        x_k_rotary = torch.einsum('bsd,sde->bse', x_k, rotary_matrix)
        
        # Apply linear transformations and split heads
        x_q = self.split_heads(self.Q(x_q_rotary))
        x_k = self.split_heads(self.K(x_k_rotary))
        x_v = self.split_heads(self.V(x_v))  # Fixed: was using self.K instead of self.V
        
        # Compute attention scores
        score = torch.matmul(x_q, x_k.transpose(-1, -2)) / math.sqrt(self.d_k)

        mask = self.CFG["attention_mask"]
        if mask is not None:
            score = score.masked_fill(mask=mask, value=-1e-9)
        
        return torch.matmul(F.softmax(score, dim=-1), x_v)
    

    def forward(self, x_q, x_k, x_v):
        x = self.attention_score(x_q, x_k, x_v)
        # Get actual dimensions from input tensor instead of using fixed config values
        batch_size, seq_len = x_q.size(0), x_q.size(1)
        print(batch_size, seq_len, self.CFG["max_sequence"])
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.CFG["emb_dim"])
        # x = x.transpose(1, 2).contiguous().view(self.CFG["batch_size"], self.CFG["max_sequence"], self.CFG["emb_dim"])

        return x



class Attention_Block(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.CFG = CONFIG
        self.rms1 = nn.RMSNorm(self.CFG["emb_dim"])
        self.rms2 = nn.RMSNorm(self.CFG["emb_dim"])
        self.ffw1 = nn.Linear(self.CFG["emb_dim"], self.CFG["ffw_hidden"])
        self.ffw2 = nn.Linear(self.CFG["ffw_hidden"], self.CFG["emb_dim"])
        self.MHA = MultiHeadAttention(self.CFG)

    def forward(self, x:torch.Tensor):
        temp_x = x.clone().detach()
        x = self.rms1(x)
        x = self.MHA(x, x, x)
        x = x + temp_x
        temp_x = x.clone().detach()
        x = self.ffw2(self.ffw1(self.rms2(x)))
        x = temp_x + x

        return x





class TestModel(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.CFG = CONFIG
        self.embed_layer = Embedder(self.CFG)
        self.att_blocks = nn.ModuleList([Attention_Block(self.CFG) for i in range(self.CFG["num_blocks"])])
        self.post_rms = nn.RMSNorm(self.CFG["emb_dim"])
        self.post_linear = nn.Linear(self.CFG["emb_dim"], self.CFG["emb_dim"])

    def forward(self, x:torch.Tensor):
        x = self.embed_layer(x)
        for layer in self.att_blocks:
            x = layer(x)
        x = F.softmax(self.post_linear(self.post_rms(x)), dim=-1)

        return x



emb_d = 512
ffw_hid = emb_d * 2

TestModel_CONFIG = {
    "batch_size": 1,
    "max_sequence": 1024,
    "emb_dim": emb_d,  
    "num_head": 2,
    "dropout_rate": 0.1,
    "num_blocks": 6,
    "input_mask": None,
    "output_mask": None,
    "vocab_size": 200019,
    "ffw_hidden": ffw_hid,
    "attention_mask": None
}




torch.manual_seed(10)
torch.set_default_dtype(torch.bfloat16)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

sentences = ["User: What is the capital of France? Assistant: The capital of France is Paris.", "User: Can you tell me about the Eiffel Tower? Assistant: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."]
tokenizer = Tokenizer(model_name='o200k_base', max_sequence=TestModel_CONFIG["max_sequence"])
# print(tokenizer.encode("<|pad|>"))
# print(tokenizer.decode([199999]))
print(f"Special token set: {tokenizer.special_tokens_set}")
print(f"Vocab size: {tokenizer.vocab_size}")
tokenized_sentences = [tokenizer.encode(sentence) for sentence in sentences]


memory = torch.cuda.memory_allocated()
model = TestModel(TestModel_CONFIG).to(device)
model_memory = torch.cuda.memory_allocated() - memory
print(f"Model memory usage: {model_memory / 1e6} Mb")
print(model)

# Forward Pass
memory = torch.cuda.memory_allocated()
outputs: torch.Tensor = model(torch.tensor(tokenized_sentences).to(device))
print(f"Llama_D model: output size: {outputs.size()}")

