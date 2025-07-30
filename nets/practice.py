import torch
import torch.nn as nn
import torch.nn.functional as F

import math

torch.manual_seed(25)

class Embedding(nn.Module):
    def __init__(self, vocab,  d_model, dr):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.dropout = nn.Dropout(dr)


    def forward(self, x):
        return self.embedding(x)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, CFG):
        pass

    def forward(self, x):
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, batch_size, sequence_length, d_model, h_num, mask=None):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.h_num = h_num
        assert self.d_model % self.h_num == 0
        self.d_k = self.d_model // self.h_num
        self.mask = mask
        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)
        self.rotary_matrix = None

    def split_head(self, x:torch.Tensor):
        return x.view(self.batch_size, self.sequence_length, self.h_num, self.d_k).transpose(1, 2).contiguous()
    
    def attention_score(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        xq = self.split_head(self.wq(q))
        xk = self.split_head(self.wk(k))
        score = xq @ xk.transpose(-1, -2)/math.sqrt(self.d_k)

        if self.mask:
            pass
        
        return F.softmax(score, dim=-1)
    
    def forward(self, q, k, v):
        score = self.attention_score(q, k, v)
        # print(f"score shape: {score.shape}, v.shape: {v.shape}")
        xv = self.split_head(self.wv(v))
        x = torch.matmul(score, xv)
        x = x.transpose(1, 2).contiguous().view(self.batch_size, self.sequence_length, self.d_model)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, batch_size, sequence_length, d_model, h_num, ffw_hid, mask=None, num_experts=4):
        super().__init__()
        self.rms1 = nn.RMSNorm(d_model)
        self.rms2 = nn.RMSNorm(d_model)
        self.mh_attention = MultiHeadAttention(batch_size, sequence_length, d_model, h_num, mask)
        self.moe = MoE(d_model, ffw_hid, num_experts)

    def forward(self, x:torch.Tensor):
        p_x = x.detach().clone()
        x = self.rms1(x)
        x = p_x + self.mh_attention(x, x, x)
        p_x = x.detach().clone()
        x = self.rms2(x)
        x = p_x + self.moe(x)

        return x


class Expert(nn.Module):
    """An expert network, which is a simple feed-forward network."""
    def __init__(self, d_model, ffw_hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffw_hid),
            nn.ReLU(),
            nn.Linear(ffw_hid, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    """A Mixture of Experts layer."""
    def __init__(self, d_model, ffw_hid, num_experts, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, ffw_hid) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: (batch_size, sequence_length, d_model)
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model) # (batch_size * sequence_length, d_model)
        
        # Get routing weights
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1) # (batch_size * sequence_length, num_experts)
        
        # Get top-k experts and their weights
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True) # Normalize weights

        # Route inputs to experts
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, d_model)

        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_weights = top_k_weights[:, i].unsqueeze(-1)

            for j in range(len(self.experts)):
                mask = (expert_indices == j)
                if mask.any():
                    selected_inputs = flat_x[mask]
                    expert_output = self.experts[j](selected_inputs)
                    final_output[mask] += expert_weights[mask] * expert_output
        
        return final_output.view(batch_size, seq_len, d_model)
    

class DemoModel(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.batch_size = CFG["batch_size"]
        self.vocab = CFG["vocab_size"]
        self.d_model = CFG["d_model"]
        self.sequence_length = CFG["sequence_length"]
        self.dropout_rate = CFG["dropout_rate"]
        self.ffw_hid = CFG["ffw_hid"]
        self.mask = CFG["mask"]
        self.num_block = CFG["num_block"]
        self.h_num = CFG["h_num"]

        self.embedding = Embedding(vocab=self.vocab, d_model=self.d_model, dr=self.dropout_rate)
        self.decoder = nn.ModuleList([
            DecoderBlock(self.batch_size, self.sequence_length, self.d_model, self.h_num, self.ffw_hid, self.mask) for i in range(self.num_block)
        ])
        self.post_rms = nn.RMSNorm(self.d_model)
        self.post_linear1 = nn.Linear(self.d_model, self.d_model)
        self.post_linear2 = nn.Linear(self.d_model, self.d_model)


    def forward(self, x):
        x = self.embedding(x)
        for decoder_block in self.decoder:
            x = decoder_block(x)
        x = F.softmax(self.post_linear2(self.post_linear1(self.post_rms(x))))

        return x




CFG = {
    "batch_size": 2,
    "vocab_size": 10000,
    "d_model": 64,
    "sequence_length": 128,
    "dropout_rate": 0.1,
    "ffw_hid": 128,
    "mask": None,
    "num_block": 6,
    "h_num": 8
}


device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(f"Using: {device}")

model = DemoModel(CFG).to(torch.bfloat16).to(device)

x = torch.randint(0, CFG["vocab_size"], (CFG["batch_size"], CFG["sequence_length"])).to(device)
print(f"x shape: {x.shape}")
output = model(x)
print(f"output shape: {output.shape}")
print(f"output: {output}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
