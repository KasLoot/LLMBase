import torch
from utils.tokenizer import get_tokenizer
import torch.nn.functional as F
from process_data import pad_token
from nets.MiniChat.model_kv_cache import MiniChat
from dataclasses import dataclass


@dataclass
class ModelConfig:
    sq_l: int = 2048
    d_model: int = 2048
    num_h: int = 16
    hidden_size: int = 4096
    drop_rate: float = 0.1
    vocab_size: int = 201088
    transformer_blocks: int = 6
    device: torch.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"



device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CFG = ModelConfig()

def inference():
    checkpoint = torch.load("/Users/liuyuxin/Documents/models/MiniChat/MiniChat-1B-pt/checkpoint_epoch_1.pt", weights_only=False, map_location=device)
    model = MiniChat(CFG=CFG, use_cache=True).to(device=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.requires_grad_(False)
    model.eval()

    with open("prompt.txt", "r") as f:
        demo_text = [f.read()]

    tokenizer = get_tokenizer()
    end_index, x = pad_token(tokenizer, demo_text)
    
    x = x[:, :CFG.sq_l].to(device=device)
    end_index = min(end_index, CFG.sq_l - 1)
    print("\n\n\n" + "-"*20 + "\n\n")

    # prefill
    with torch.no_grad():
        logits = model(x[:, :end_index], start_pos=0)
        logits_for_next_token = logits[0, -1, :]
        pred_token = torch.argmax(logits_for_next_token, dim=-1)
        x[0, end_index] = pred_token.item()
        pred_text = tokenizer.decode([pred_token.item()])
        print(pred_text, end="", flush=True)


    # generation
    with torch.no_grad():
        for i in range(end_index + 1, CFG.sq_l):
            logits = model(x[:, i-1:i], start_pos=i-1)
            
            logits_for_next_token = logits[0, -1, :]
            
            pred_token = torch.argmax(logits_for_next_token, dim=-1)
            
            x[0, i] = pred_token.item()
            
            pred_text = tokenizer.decode([pred_token.item()])
            print(pred_text, end="", flush=True)





inference()