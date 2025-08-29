import torch
from tqdm import tqdm
from nets.MiniChat.model import ModelConfig

CFG = ModelConfig()


def pad_token(tokenizer, token) -> torch.Tensor:
    max_sq = CFG.sq_l
    end_index = 0

    padded_tokens = torch.full((len(token), max_sq), 200000)
    start_tensor = torch.full((len(token), 1), 199998)
    padded_tokens = torch.hstack((start_tensor, padded_tokens))



    idx = 0
    for sample in tqdm(token):

        tk = tokenizer.encode(sample, allowed_special="all") + [199999]
        print(len(tk))
        if len(tk) > max_sq + 1:
            tk = tk[:max_sq + 1]
            padded_tokens[idx, 1:] = torch.tensor(tk)[:max_sq] # shrink tokens
            end_index = max_sq - 1
        else:
            padded_tokens[idx, 1:1+len(tk)] = torch.tensor(tk) # copy tokens to padded tokens
            end_index = len(tk)
        
        idx += 1

    return end_index, padded_tokens


def to_structured_data(data):
    pass
    