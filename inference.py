import torch
from utils.tokenizer import get_tokenizer
import torch.nn.functional as F
from utils.dataset import LanguageModelDataset, pad_token
from nets.demo_model.model import MiniChat


from torch.utils.data import Dataset, DataLoader
from nets.demo_model.model import ModelConfig


device = "cuda" if torch.cuda.is_available() else "cpu"
CFG = ModelConfig()

def inference():
    checkpoint = torch.load("", weights_only=True)
    model = MiniChat(CFG=CFG).to(device=device)
    model.load_state_dict(checkpoint[""])
    model.eval()

    demo_text = ["Lesson 1: How to"]

    # dataset = LanguageModelDataset(path="/home/yuxin/datasets/LLMBase/train_tiny_textbook.pt")

    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # x, _ = next(iter(data_loader))


    tokenizer = get_tokenizer()
    end_index, x = pad_token(tokenizer, demo_text)
        
    x = x.to(device=device)
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