import torch
from torch.utils.data import DataLoader
import wandb
import os
from datetime import datetime


from nets.MiniChat_Vision.model import ModelConfig, MiniChat_Vision

from utils.dataset import VLM_SFT_Dataset

from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
CFG = ModelConfig()



def custom_collate_fn(batch):
    """Custom collate function to handle variable-length assistant_indexes"""
    text_tokens = torch.stack([item[0] for item in batch])
    image_embeds = torch.stack([item[1] for item in batch])
    assistant_indexes = [item[2] for item in batch]  # Keep as list of lists
    return text_tokens, image_embeds, assistant_indexes







def save_model(model, optimizer, epoch, loss, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'model_config': CFG.__dict__  # Save model configuration
    }

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

    if is_best:
        print(f"New best model saved with loss: {loss:.4f}")


def load_model(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')


def train():
    # Initialize wandb
    wandb.init(
        project="llmbase-training",
        config={
            "learning_rate": 1e-4,
            "epochs": 20,
            "batch_size": 16,
            "sequence_length": CFG.sq_l,
            "d_model": CFG.d_model,
            "num_heads": CFG.num_h,
            "hidden_size": CFG.hidden_size,
            "vocab_size": CFG.vocab_size,
            "transformer_blocks": CFG.transformer_blocks,
            "drop_rate": CFG.drop_rate,
        }
    )

    # Create save directories
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_model_path = os.path.join(save_dir, f"latest_model_{timestamp}.pt")
    best_model_path = os.path.join(save_dir, f"best_model_{timestamp}.pt")
    
    # Log model paths to wandb config
    wandb.config.update({
        "latest_model_path": latest_model_path,
        "best_model_path": best_model_path,
        "save_directory": save_dir
    })
    
    # Initialize best loss tracker
    best_loss = float('inf')




    dataset = VLM_SFT_Dataset(
    text_tokens_path="/home/yuxin/datasets/BUAADreamer/llava-en-zh-2k/BUAADreamer_processed/file_0_texttokens.pt",
    image_embeds_path="/home/yuxin/datasets/BUAADreamer/llava-en-zh-2k/BUAADreamer_processed/file_0_image_embeds.pt",
    assistant_index_path="/home/yuxin/datasets/BUAADreamer/llava-en-zh-2k/BUAADreamer_processed/file_0_assistant_indexes.json",
    seq_length=2048
)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)



    prev_memory = torch.cuda.memory_allocated()
    model = MiniChat_Vision(CFG=CFG).to(device=device).to(torch.bfloat16).train()
    memory_after = torch.cuda.memory_allocated()
    print(f"total memory use for loading model: {(memory_after - prev_memory)/1e6} Mb, {(memory_after - prev_memory)/1e9} Gb")

    ckp = torch.load("/home/yuxin/models/LLMBase/MiniChat/MiniChat-1B-pt-v1.pt", map_location=device)
    model.load_state_dict(ckp['model_state_dict'], strict=False)
    print("Loaded pretrained MiniChat weights into MiniChat_Vision model.")

    # Log memory usage to wandb
    wandb.log({
        "model_memory_mb": (memory_after - prev_memory)/1e6,
        "model_memory_gb": (memory_after - prev_memory)/1e9
    })

    total_params = 0

    for parameter in model.parameters():
        total_params += parameter.numel()

    print(f"model parameters: {total_params/1e9} B")
    
    # Log model parameters to wandb
    wandb.log({"model_parameters_billions": total_params/1e9})

    learning_rate = 1e-4
    optmizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=200000)

    step = 0
    for epoch in tqdm(range(3)):
        epoch_loss = 0
        batch_count = 0
        
        for batch in tqdm(dataloader):
            optmizer.zero_grad()
            inputs, image_embeds, targets = batch
            # input: (batch_size, seq_length)
            # image_embeds: (batch_size, 256, 1024)
            # target: (batch_size, seq_length), with tokens before last <assistant> set to 200000 (ignore_index)
            inputs: torch.Tensor = inputs.to(device)
            image_embeds: torch.Tensor = image_embeds.to(device)
            targets: torch.Tensor = targets.to(device)
            logits: torch.Tensor = model(image_embeds, inputs)  # Raw logits from model
            logits = logits.contiguous().view(-1, CFG.vocab_size)
            targets = targets.view(-1).long()
            loss = loss_func(logits, targets)
            loss.backward()
            optmizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            step += 1
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

            break
        break
            


if __name__ == "__main__":
    train()


