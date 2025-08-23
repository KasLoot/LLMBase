import torch
from utils.tokenizer import get_tokenizer
import torch.nn.functional as F
from process_data import pad_token
from nets.MiniChat.model import MiniChat
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
    checkpoint = torch.load("/Users/liuyuxin/Documents/models/MiniChat/MiniChat-1B-pt/MiniChat-1B-pt.pt", weights_only=True, map_location=device)
    model = MiniChat(CFG=CFG).to(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.eval()

    demo_text = ["Open new Terminal windows and tabs on Mac\nYou can open new Terminal windows and tabs with the default profile, the same profile used by the active window or tab, or a profile you specify.\nUse General settings to choose the profiles used when opening new windows and tabs. If you haven’t chosen a default profile, the Basic profile is used (see Specify default and startup Terminal window profiles)."]

    # dataset = LanguageModelDataset(path="/home/yuxin/datasets/LLMBase/train_tiny_textbook.pt")

    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # x, _ = next(iter(data_loader))


    tokenizer = get_tokenizer()
    end_index, x = pad_token(tokenizer, demo_text)
    
    # The pad_token function returns tensors of shape [batch_size, sq_l + 1] 
    # but model expects [batch_size, sq_l], so we slice appropriately
    x = x[:, :CFG.sq_l].to(device=device)  # Take only the first sq_l tokens
    end_index = min(end_index, CFG.sq_l - 1)  # Adjust end_index accordingly
    # predict from end_index + 1 position
    with torch.no_grad():
        for i in range(end_index, CFG.sq_l):
            # 1. Get model logits.
            # Note: For efficiency, this should ideally use a KV cache.
            logits = model(x)
            
            # 2. Get logits for the specific token we want to predict.
            # To predict the token at position `i`, we use the output from position `i-1`.
            # Logits shape: (batch_size, sequence_length, vocab_size)
            logits_for_next_token = logits[0, i - 1, :] # Shape: (vocab_size)
            
            # 3. Get the most likely token (the prediction).
            pred_token = torch.argmax(logits_for_next_token, dim=-1)
            
            # 4. CRITICAL: Update the input tensor `x` with the predicted token.
            # This is how the model knows what it generated in the previous step.
            x[0, i] = pred_token.item()
            
            # (Optional but recommended) Stop if an end-of-sequence token is generated.
            # if pred_token.item() == tokenizer.eos_token_id:
            #     break
                
            # 5. Decode and print the result.
            pred_text = tokenizer.decode([pred_token.item()])
            print(pred_text, end="", flush=True)





inference()