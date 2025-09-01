import torch
import torch.nn as nn


import torch.nn.functional as F
import torch.distributed as dist

import math

from dataclasses import dataclass

@dataclass
class ModelConfig:
    sq_l: int = 200
    d_model: int = 64
    num_h: int = 16
    hidden_size: int = 128
    drop_rate: float = 0.1
    vocab_size: int = 201088
    transformer_blocks: int = 2
    num_experts: int = 4
    top_k = 2
    device: torch.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    world_size = torch.cuda.device_count()


class Embedding(nn.Module):
    def __init__(self, CFG: ModelConfig):
        super().__init__()
        self.vocab = CFG.vocab_size
        self.d_model = CFG.d_model
        self.dr = CFG.drop_rate
        self.emb = nn.Embedding(self.vocab, self.d_model)
        self.drop = nn.Dropout(self.dr)

    def forward(self, x: torch.Tensor):
        x = self.drop(self.emb(x))
        return x
    


def rotary_positional_embedding(x, freqs_cos, freqs_sin):
    """
    Apply rotary position embedding to input tensor x.
    
    Args:
        x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        freqs_cos: Cosine frequencies of shape (seq_len, head_dim // 2)
        freqs_sin: Sine frequencies of shape (seq_len, head_dim // 2)
    
    Returns:
        Tensor with rotary position embedding applied
    """
    # Split x into first half and second half along the last dimension
    x1, x2 = x.chunk(2, dim=-1)
    
    # Apply rotation: (x1, x2) -> (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
    rotated = torch.cat([
        x1 * freqs_cos - x2 * freqs_sin,
        x1 * freqs_sin + x2 * freqs_cos
    ], dim=-1)
    
    return rotated


def precompute_rope_freqs(seq_len, d_k, base=10000, device="cpu"):
    """
    Precompute the cosine and sine frequencies for rotary position embedding.
    
    Args:
        seq_len: Maximum sequence length
        d_k: Head dimension
        base: Base for frequency computation
        device: Device to place tensors on
    
    Returns:
        freqs_cos, freqs_sin: Precomputed cosine and sine frequencies
    """
    # Create position indices
    positions = torch.arange(seq_len, device=device).float()
    
    # Create frequency indices
    freqs = 1.0 / (base ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
    
    # Outer product to get all position-frequency combinations
    freqs = torch.outer(positions, freqs)
    
    # Compute cos and sin
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    
    return freqs_cos, freqs_sin


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class MHAttention(nn.Module):
    def __init__(self, CFG:ModelConfig):
        super().__init__()
        self.d_model = CFG.d_model
        self.num_h = CFG.num_h
        self.sq_l = CFG.sq_l
        self.device = CFG.device

        assert self.d_model % self.num_h == 0, "self.d_model % self.num_h != 0"
        self.d_k = self.d_model // self.num_h
        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)
        self.wo = nn.Linear(self.d_model, self.d_model)
        
        # Precompute RoPE frequencies
        self.freqs_cos, self.freqs_sin = precompute_rope_freqs(
            self.sq_l, self.d_k, device=self.device
        )
        # Register as buffers so they move with the model
        self.register_buffer('rope_cos', self.freqs_cos)
        self.register_buffer('rope_sin', self.freqs_sin)

    def split_head(self, x:torch.Tensor):
        bs, sq_l = x.shape[0], x.shape[1]
        x = x.reshape(bs, sq_l, self.num_h, self.d_k).transpose(1, 2).contiguous()
        return x
    
    def attention_score(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        score = (q @ k.transpose(-1, -2))/math.sqrt(self.d_k)
        mask = torch.tril(torch.ones(q.shape[2], q.shape[2], device=q.device)).unsqueeze(0).unsqueeze(0)

        if mask is not None:
            score = score.masked_fill(mask==0, torch.finfo(score.dtype).min)

        score = F.softmax(score, dim=-1) @ v

        return score
    
    def forward(self, xq, xk, xv):
        q = self.split_head(self.wq(xq))
        k = self.split_head(self.wk(xk))
        v = self.split_head(self.wv(xv))

        # Get sequence length for current input
        seq_len = q.shape[2]
        
        # Apply rotary position embedding to queries and keys
        # Slice the precomputed frequencies to match current sequence length
        freqs_cos = self.rope_cos[:seq_len, :]
        freqs_sin = self.rope_sin[:seq_len, :]
        
        # Apply RoPE to queries and keys
        q = rotary_positional_embedding(q, freqs_cos, freqs_sin)
        k = rotary_positional_embedding(k, freqs_cos, freqs_sin)

        output = self.attention_score(q, k, v)
        bs, _, sq_l, _ = output.shape

        output = output.transpose(1, 2).contiguous().reshape(bs, sq_l, self.d_model)

        output = self.wo(output)

        return output



class MOE(nn.Module):
    def __init__(self, CFG:ModelConfig):
        super().__init__()
        self.num_experts = CFG.num_experts
        self.d_model = CFG.d_model
        self.hidden_size = CFG.hidden_size
        self.world_size = CFG.world_size
        self.top_k = CFG.top_k
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.mlp1_w = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.d_model))
        self.mlp1_b = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.mlp2_w = nn.Parameter(torch.empty(self.num_experts, self.d_model, self.hidden_size))
        self.mlp2_b = nn.Parameter(torch.empty(self.num_experts, self.d_model))
        nn.init.xavier_uniform_(self.mlp1_w)
        nn.init.xavier_uniform_(self.mlp2_w)


    def forward(self, x: torch.Tensor):
        gate = self.gate(x)

        # experts = torch.topk(gate, k=2, dim=-1)
        # experts_weights = F.softmax(experts.values, dim=-1)
        # experts_indices = experts.indices

        # print(f"experts_weights: {experts_weights.shape}")
        # print(f"expert_indices: {experts_indices.shape}\n{experts_indices[:,:1,:]}")
        

        # print(f"self.mlp1_W shape: {self.mlp1_w.shape}")

        # mlp1_w = self.mlp1_w[experts_indices, ...]
        # mlp1_b = self.mlp1_b[experts_indices, ...]
        # mlp2_w = self.mlp2_w[experts_indices, ...]
        # mlp2_b = self.mlp2_b[experts_indices, ...]

        # print(f"x shape: {x.shape}")
        # print(f"mlp1_w shape: {mlp1_w.shape}")
        # print(f"mlp1_b shape: {mlp1_b.shape}")
        # print(f"mlp2_w shape: {mlp2_w.shape}")
        # print(f"mlp2_b shape: {mlp2_b.shape}")

        # t = torch.einsum("bsehd,bsd->bseh", mlp1_w, x) + mlp1_b
        # print(f"t shape: {t.shape}")

        # t = F.silu(t)
        # print(f"t shape (swiglu): {t.shape}")

        # t = torch.einsum("bsedh,bseh->bsed", mlp2_w, t) + mlp2_b
        # print(f"t shape: {t.shape}")
        # t = torch.einsum("bsed,bse->bsd", t, experts_weights)
        # print(f"t shape: {t.shape}")

        # t = t + x

        # return t
    


        # Find top 2 experts for each token
        weights, indices = torch.topk(gate, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1) # bse (e=2)
        
        # Gather the weights for the chosen experts
        # mlp1_w: (num_experts, hidden_size, d_model) -> nhd
        # mlp1_b: (num_experts, hidden_size) -> nh
        # mlp2_w: (num_experts, d_model, hidden_size) -> ndh
        # mlp2_b: (num_experts, d_model) -> nd
        
        # indices: bse
        # x: bsd
        
        # einsum approach
        # expand x for experts
        x_expanded = x.unsqueeze(2).expand(-1, -1, 2, -1) # bsed
        
        # Get expert weights
        w1 = self.mlp1_w[indices] # bsehd
        b1 = self.mlp1_b[indices] # bseh
        w2 = self.mlp2_w[indices] # bsedh
        b2 = self.mlp2_b[indices] # bsed

        print(f"x shape: {x.shape}")
        print(f"x_expanded shape: {x_expanded.shape}")
        print(f"mlp1_w shape: {w1.shape}")
        print(f"mlp1_b shape: {b1.shape}")
        print(f"mlp2_w shape: {w2.shape}")
        print(f"mlp2_b shape: {b2.shape}")


        # First MLP layer
        hidden = torch.einsum('bsed,bsehd->bseh', x_expanded, w1) + b1
        hidden = F.silu(hidden)
        
        # Second MLP layer
        out = torch.einsum('bseh,bsedh->bsed', hidden, w2) + b2
        
        # Weight the outputs
        weighted_out = torch.einsum('bsed,bse->bsd', out, weights)
        
        return weighted_out



class TransformerBlock(nn.Module):
    def __init__(self, CFG: ModelConfig):
        super().__init__()
        self.d_model = CFG.d_model
        self.hidden_size = CFG.hidden_size
        self.dr = CFG.drop_rate
        self.rms1 = nn.RMSNorm(self.d_model)
        self.mhAttention = MHAttention(CFG=CFG)
        self.rms2 = nn.RMSNorm(self.d_model)
        self.moe = MOE(CFG=CFG)

    def forward(self, x: torch.Tensor):
        out = self.rms1(x)
        out = self.mhAttention(out, out, out)
        x = out + x
        out = self.rms2(x)
        out = self.moe(out)
        x = out + x

        return x


class Transformer(nn.Module):
    def __init__(self, CFG: ModelConfig):
        super().__init__()
        self.trans_blocks = CFG.transformer_blocks
        self.transformer = nn.ModuleList([TransformerBlock(CFG=CFG) for _ in range(self.trans_blocks)])

    def forward(self, x: torch.Tensor):
        for block in self.transformer:
            x = block(x)

        return x


class MiniChat_MoE(nn.Module):
    def __init__(self, CFG: ModelConfig):
        super().__init__()
        self.d_model = CFG.d_model
        self.vocab = CFG.vocab_size
        self.embedding = Embedding(CFG=CFG)
        self.transformer = Transformer(CFG=CFG)

        self.post_rms = nn.RMSNorm(self.d_model)
        self.linear = nn.Linear(self.d_model, self.vocab)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.post_rms(x)
        x = self.linear(x)

        return x
    



if __name__ == "__main__":
    x = torch.randint(0, 201088, (3, 200)).cuda()
    model = MiniChat_MoE(CFG=ModelConfig()).cuda()
    output = model(x)