import torch
from torch.utils.data import Dataset, DataLoader
import wandb
import os
from datetime import datetime

from nets.MiniChat.model import MiniChat

from nets.MiniChat.model import ModelConfig

from utils.dataset import LanguageModelDataset

from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
CFG = ModelConfig()


def save_model(model, optimizer, epoch, loss, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
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




    dataset = LanguageModelDataset(path="/home/yuxin/datasets/LLMBase/train_tiny_textbook.pt")

    train_data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)


    # print(f"input shape: {inputs.shape}, contigous: {inputs.is_contiguous()}\n{inputs[0]}")
    # print(f"targets shape: {targets.shape}, contigous: {targets.is_contiguous()}\n{targets[0]}")

    
    prev_momory = torch.cuda.memory_allocated()
    model = MiniChat(CFG=CFG).to(device=device).to(torch.bfloat16).train()
    momory_after = torch.cuda.memory_allocated()
    print(f"total memory use for loading model: {(momory_after - prev_momory)/1e6} Mb, {(momory_after - prev_momory)/1e9} Gb")
    
    # Log memory usage to wandb
    wandb.log({
        "model_memory_mb": (momory_after - prev_momory)/1e6,
        "model_memory_gb": (momory_after - prev_momory)/1e9
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
    for epoch in tqdm(range(20)):
        epoch_loss = 0
        batch_count = 0
        
        for batch in tqdm(train_data_loader):
            inputs, targets = batch
            inputs: torch.Tensor = inputs.to(device)
            targets: torch.Tensor = targets.to(device)
            optmizer.zero_grad()
            logits: torch.Tensor = model(inputs)  # Raw logits from model
            logits = logits.contiguous().view(-1, CFG.vocab_size)
            targets = targets.view(-1).long()

            loss = loss_func(logits, targets)
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            
            # Log metrics to wandb
            current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            wandb.log({
                "loss": loss.item(),
                "epoch": epoch + 1,
                "step": step,
                "learning_rate": learning_rate,
                "gpu_memory_allocated_mb": current_memory / 1e6 if torch.cuda.is_available() else 0
            })
            
            epoch_loss += loss.item()
            batch_count += 1
            step += 1
            
            loss.backward()
            optmizer.step()
        
        # Log epoch-level metrics
        avg_epoch_loss = epoch_loss / batch_count
        wandb.log({
            "epoch_avg_loss": avg_epoch_loss,
            "epoch": epoch + 1
        })
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save latest model
        save_model(model, optmizer, epoch + 1, avg_epoch_loss, latest_model_path)
        
        # Save best model if this is the best loss so far
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_model(model, optmizer, epoch + 1, avg_epoch_loss, best_model_path, is_best=True)
            
            # Log best model info to wandb
            wandb.log({
                "best_loss": best_loss,
                "best_epoch": epoch + 1
            })

    # Training completed - save final summary
    print(f"\nTraining completed!")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Latest model saved to: {latest_model_path}")
    print(f"Best model saved to: {best_model_path}")
    
    # Log final metrics to wandb
    wandb.log({
        "final_best_loss": best_loss,
        "training_completed": True
    })

    # Finish wandb run
    wandb.finish()

    



train()





# model = DemoTextModel(CFG).to(device).to(torch.bfloat16)
# x = torch.randint(0, 100000, (CFG["batch_size"], CFG["sq_l"])).to(device)


# mem_p = torch.cuda.memory_allocated()
# y = model(x)
# peak_mem = torch.cuda.max_memory_allocated()
# print(f"memory usage: {(peak_mem - mem_p)/1e6} Mb")