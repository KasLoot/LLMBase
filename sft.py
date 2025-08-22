import torch
from torch.utils.data import DataLoader
import wandb
import os
from datetime import datetime

from nets.MiniChat.model import MiniChat

from nets.MiniChat.model import ModelConfig

from utils.dataset import LanguageModel_SFT_Dataset

from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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




def train():
    """
    Full parameter training loop for fine-tuning the model.
    """

    # Create save directories
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_model_path = os.path.join(save_dir, f"latest_model_{timestamp}.pt")
    best_model_path = os.path.join(save_dir, f"best_model_{timestamp}.pt")
    
    
    # Initialize best loss tracker
    best_loss = float('inf')


    dataset = LanguageModel_SFT_Dataset(path="/Users/liuyuxin/Documents/LLMBase/data_samples/sft_data.pt")

    train_data_loader = DataLoader(dataset, batch_size=2, shuffle=True)


    # print(f"input shape: {inputs.shape}, contigous: {inputs.is_contiguous()}\n{inputs[0]}")
    # print(f"targets shape: {targets.shape}, contigous: {targets.is_contiguous()}\n{targets[0]}")

    
    prev_momory = torch.cuda.memory_allocated()
    model = MiniChat(CFG=CFG).to(device=device).to(torch.bfloat16).train()
    momory_after = torch.cuda.memory_allocated()
    print(f"total memory use for loading model: {(momory_after - prev_momory)/1e6} Mb, {(momory_after - prev_momory)/1e9} Gb")
    

    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
    print(f"model parameters: {total_params/1e9} B")
    

    learning_rate = 1e-4
    optmizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=200000)

    step = 0
    for epoch in tqdm(range(1)):
        epoch_loss = 0
        batch_count = 0
        
        for batch in tqdm(train_data_loader):
            inputs, targets = batch
            inputs: torch.Tensor = inputs.to(device)
            targets: torch.Tensor = targets.to(device)
            # Find where assistant response starts (after the assistant marker)
            target_start_index = (inputs == 200014).nonzero(as_tuple=True)[1]
            optmizer.zero_grad()
            logits: torch.Tensor = model(inputs)  # Raw logits from model
            print(f"logits shape: {logits.shape}")
            
            # Properly mask targets: set non-assistant tokens to ignore_index
            for i in range(logits.shape[0]):
                # Mask everything before the assistant response (keep assistant marker, mask response tokens)
                # Assistant response starts AFTER the assistant marker
                mask_end = target_start_index[i] + 1  # +1 to include the assistant marker in masking
                targets[i, :mask_end] = 200000  # Use ignore_index, not 0
                
            # Don't modify logits - let the model see the full context

            print(f"logits sample: {logits[0, :2]}")
            print(f"target sample: {targets[0, :2]}")
            """
            logits sample: tensor([[0., 0., 0.,  ..., 0., 0., 0.], [0., 0., 0.,  ..., 0., 0., 0.]], device='mps:0', dtype=torch.bfloat16, grad_fn=<SliceBackward0>)
            target sample: tensor([0, 0], device='mps:0')
            """

            logits = logits.contiguous().view(-1, CFG.vocab_size)
            print(f"logits shape: {logits.shape}")

            

            targets = targets.view(-1).long()
            print(f"targets shape: {targets.shape}")
            loss = loss_func(logits, targets)
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

            
            epoch_loss += loss.item()
            batch_count += 1
            step += 1
            
            loss.backward()
            optmizer.step()

        # Log epoch-level metrics
        avg_epoch_loss = epoch_loss / batch_count

        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save latest model
        save_model(model, optmizer, epoch + 1, avg_epoch_loss, latest_model_path)
        
        # Save best model if this is the best loss so far
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_model(model, optmizer, epoch + 1, avg_epoch_loss, best_model_path, is_best=True)
            
            # Log best model info to wandb

    # Training completed - save final summary
    print(f"\nTraining completed!")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Latest model saved to: {latest_model_path}")
    print(f"Best model saved to: {best_model_path}")
    
    # Log final metrics to wandb



    



train()





# model = DemoTextModel(CFG).to(device).to(torch.bfloat16)
# x = torch.randint(0, 100000, (CFG["batch_size"], CFG["sq_l"])).to(device)


# mem_p = torch.cuda.memory_allocated()
# y = model(x)
# peak_mem = torch.cuda.max_memory_allocated()
# print(f"memory usage: {(peak_mem - mem_p)/1e6} Mb")