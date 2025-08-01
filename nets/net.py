import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EmbeddingModel(nn.Module):
    def __init__(self, emb_dim, vocab_size, dropout_rate=0.1):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x


class Positional_Embedding(nn.Module):
    def __init__(self, emb_dim, max_sequence, dropout_rate=0.1):
        super(Positional_Embedding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        position = torch.arange(0, max_sequence, dtype=torch.bfloat16).unsqueeze(1) # This dtype is not affected by the global dtype setting
        self.position_encoding = torch.zeros(max_sequence, emb_dim)
        div_term = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.bfloat16)* -math.log(10000.0)/emb_dim)

        # Apply sin to even indices (0, 2, 4, ...)
        self.position_encoding[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices (1, 3, 5, ...)
        self.position_encoding[:, 1::2] = torch.cos(position * div_term)

        self.position_encoding = self.position_encoding.unsqueeze(0)

        # Register 'pe' as a buffer. Buffers are part of the model's state,
        # but they are not considered model parameters to be trained.
        self.register_buffer('position_encoding_buffer', self.position_encoding)

    def forward(self, x):
        x = self.dropout(x + self.position_encoding_buffer[:, :x.size(1)])
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_head, mask=None):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.d_k = emb_dim // num_head
        self.mask = mask
        self.Q = nn.Linear(self.emb_dim, self.emb_dim)
        self.K = nn.Linear(self.emb_dim, self.emb_dim)
        self.V = nn.Linear(self.emb_dim, self.emb_dim)

    def split_head(self, x: torch.Tensor):
        B, S, D = x.size()
        assert self.emb_dim == D, "unmatch embedding dimention"
        x = x.view(B, S, self.num_head, self.d_k).transpose(1, 2)
        return x

    def attention_score(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        score = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k))
        if (self.mask) is not None:
            score = score.masked_fill(mask=self.mask, value=-1e9)
        score = torch.matmul(F.softmax(score, dim=-1), V)
        return score

    def concat(self, score: torch.Tensor):
        B, NH, S, DK = score.size()
        return score.transpose(1, 2).contiguous().view(B, S, NH*DK)


    def forward(self, q, k, v):
        Q = self.split_head(self.Q(q))
        K = self.split_head(self.K(k))
        V = self.split_head(self.V(v))
        score = self.concat(self.attention_score(Q, K, V))
        return score


class FFN(nn.Module):
    def __init__(self, emb_dim):
        super(FFN, self).__init__()
        self.emb_dim = emb_dim
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.ffn(x)

class Add_Norm(nn.Module):
    def __init__(self, emb_dim, dropout_rate=0.1):
        super(Add_Norm, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, previous_x, x):
        x = self.dropout(x) # Drop out neurons for regularization. This can prevent model to learn noise from the dataset.
        x = previous_x + x
        x = self.layer_norm(x)
        return x


class Encoder_Block(nn.Module):
    def __init__(self, emb_dim, num_head, mask, dropout_rate):
        super(Encoder_Block, self).__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.mh_attention = MultiHeadAttention(self.emb_dim, self.num_head, mask)
        self.add_norm1 = Add_Norm(self.emb_dim, dropout_rate)
        self.add_norm2 = Add_Norm(self.emb_dim, dropout_rate)
        self.ffn = FFN(self.emb_dim)
        
    def forward(self, x):
        attention = self.mh_attention(x, x, x)
        add_norm = self.add_norm1(x, attention)
        ff = self.ffn(add_norm)
        x = self.add_norm2(add_norm, ff)
        return x

class Encoder(nn.Module):
    def __init__(self, emb_dim, num_head, mask, num_block, dropout_rate):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.encoder = nn.ModuleList([Encoder_Block(self.emb_dim, self.num_head, mask, dropout_rate) for _ in range(num_block)])

    def forward(self, x):
        for encoder_block in self.encoder:
            x = encoder_block(x)   
        return x


class Decoder_Block(nn.Module):
    def __init__(self, emb_d, num_heads, mask, dropout_rate):
        super(Decoder_Block, self).__init__()
        self.emb_d = emb_d
        self.num_heads = num_heads
        self.masked_mh_attention = MultiHeadAttention(self.emb_d, self.num_heads, mask)
        self.add_norm1 = Add_Norm(self.emb_d, dropout_rate)
        self.mh_attention = MultiHeadAttention(self.emb_d, self.num_heads)
        self.add_norm2 = Add_Norm(self.emb_d, dropout_rate)
        self.ffn = FFN(self.emb_d)
        self.add_norm3 = Add_Norm(self.emb_d, dropout_rate)
    
    def forward(self, encoder_output, tgt_emb):
        self_attention = self.masked_mh_attention(tgt_emb, tgt_emb, tgt_emb)
        tgt_emb = self.add_norm1(tgt_emb, self_attention)
        cross_attention = self.mh_attention(tgt_emb, encoder_output, encoder_output)
        tgt_emb = self.add_norm2(tgt_emb, cross_attention)
        tgt_emb = self.add_norm3(tgt_emb, self.ffn(tgt_emb))
        return tgt_emb


class Decoder(nn.Module):
    def __init__(self, emb_dim, num_head, mask, num_block, dropout_rate):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList([Decoder_Block(emb_dim, num_head, mask, dropout_rate) for _ in range(num_block)])

    def forward(self, encoder_output, tgt_emb):
        for decoder_block in self.decoder:
            tgt_emb = decoder_block(encoder_output, tgt_emb)
        return tgt_emb

    
class Transformer_Block(nn.Module):
    def __init__(self, emb_dim, num_head, max_sequence, dropout_rate, enc_num_block, decoder_num_block, input_mask, output_mask, vocab_size):
        super(Transformer_Block, self).__init__()
        self.position_emb = Positional_Embedding(emb_dim, max_sequence, dropout_rate)
        self.encoder = Encoder(emb_dim, num_head, input_mask, enc_num_block, dropout_rate)
        self.decoder = Decoder(emb_dim, num_head, output_mask, decoder_num_block, dropout_rate)
        self.linear = nn.Linear(emb_dim, vocab_size)
        # self.softmax = nn.Softmax(-1)

    def forward(self, input_emb, tgt_emb):
        input_emb = self.position_emb(input_emb)
        tgt_emb = self.position_emb(tgt_emb)
        input_emb = self.encoder(input_emb)
        out_emb = self.decoder(input_emb, tgt_emb)
        # out_prob = self.softmax(self.linear(tgt_emb))

        return out_emb




class DemoModel(nn.Module):
    def __init__(self, emb_dim, num_head, max_sequence, dropout_rate, enc_num_block, decoder_num_block, input_mask, output_mask, vocab_size):
        super(DemoModel, self).__init__()
        self.embedding_model = EmbeddingModel(emb_dim, vocab_size, dropout_rate)
        self.transformer_block = Transformer_Block(emb_dim, num_head, max_sequence, dropout_rate, enc_num_block, decoder_num_block, input_mask, output_mask, vocab_size)

    def forward(self, input_ids, tgt_ids):
        input_emb = self.embedding_model(input_ids)
        tgt_emb = self.embedding_model(tgt_ids)
        output = self.transformer_block(input_emb, tgt_emb)
        return output

