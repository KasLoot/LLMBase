import torch
import torch.nn.functional as F

import time

from nets.MiniChat.model import MiniChat, MiniChat_sliding, ModelConfig


def analyse(model:torch.nn.Module):
    CFG = ModelConfig()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    m1 = torch.cuda.memory_allocated() if device == "cuda" else torch.mps.current_allocated_memory() if device == "mps" else exit(0)
    model = model.to(device=device).to(torch.bfloat16)
    m2 = torch.cuda.memory_allocated() if device == "cuda" else torch.mps.current_allocated_memory() if device == "mps" else exit(0)
    print(f"Model size: {(m2 - m1) / 1024 / 1024:.2f} MB")

    dummy_input = torch.randint(0, CFG.vocab_size, (1, CFG.sq_l), device=device).to(device=device)

    m3 = torch.cuda.memory_allocated() if device == "cuda" else torch.mps.current_allocated_memory() if device == "mps" else exit(0)
    output = model(dummy_input)
    m4 = torch.cuda.memory_stats()["allocated_bytes.all.peak"] if device == "cuda" else torch.mps.current_allocated_memory() if device == "mps" else exit(0)
    print(f"process memory peak usage: {(m4 - m3) / 1024 / 1024:.2f} MB")



analyse(MiniChat_sliding(ModelConfig()))
analyse(MiniChat(ModelConfig()))