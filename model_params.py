import torch
from nets.MiniChat.model import MiniChat, ModelConfig

model_config = ModelConfig()
model = MiniChat(model_config)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Parameters: {params}")