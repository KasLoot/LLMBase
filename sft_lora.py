import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import wandb
import os
from datetime import datetime

from nets.MiniChat.model import MiniChat, ModelConfig

from utils.dataset import LanguageModel_SFT_Dataset

from tqdm import tqdm

CFG = ModelConfig()


class LoRALinear(nn.Module):
    """
    A wrapper for a linear layer that adds a LoRA update.
    """
    def __init__(self, linear_layer: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha

        # self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_A = nn.Linear(self.in_features, self.rank)
        self.lora_B = nn.Linear(self.rank, self.out_features)
        self.scaling = self.alpha / self.rank

        

    def forward(self, x):
        """
        Forward pass for the LoRA-wrapped linear layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the original layer plus the LoRA update.
        """
        original_output = self.linear(x)
        
        # More efficient implementation:
        # Pass the input through the low-rank matrices sequentially
        # instead of forming the full lora_update matrix.
        # x -> (A*x) -> B*(A*x)
        lora_A_output = self.lora_A(x)
        lora_B_output = self.lora_B(lora_A_output)
        
        lora_output = lora_B_output * self.scaling
        
        return original_output + lora_output





def apply_lora_to_model(model, rank=8, alpha=16):
    """
    Applies LoRA to the linear layers of the multi-head attention blocks in the model.
    The original weights of the linear layers are preserved.

    Args:
        model (nn.Module): The model to modify.
        rank (int): The rank for the LoRA matrices.
        alpha (int): The alpha scaling factor for LoRA.
    """
    for transformer_block in model.transformer.transformer:
        attention = transformer_block.mhAttention
        
        # By passing the original layer (e.g., attention.wq) to the LoRALinear constructor,
        # we ensure that the LoRALinear wrapper contains the original, pre-trained weights.
        # The forward pass of LoRALinear then adds the low-rank update to the output
        # of the original frozen layer, thus not overriding the pre-trained weights.
        attention.wq = LoRALinear(attention.wq, rank=rank, alpha=alpha)
        attention.wk = LoRALinear(attention.wk, rank=rank, alpha=alpha)
        attention.wv = LoRALinear(attention.wv, rank=rank, alpha=alpha)
        attention.wo = LoRALinear(attention.wo, rank=rank, alpha=alpha)


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

def get_lora_state_dict(model):
    """
    Filters the model's state_dict to include only the trainable (LoRA) parameters.
    """
    return {k: v for k, v in model.state_dict().items() if v.requires_grad}


def save_lora_model(model, optimizer, epoch, loss, save_path, is_best=False):
    """Save model checkpoint, now saving only LoRA parameters."""
    
    # Get only the trainable LoRA parameters
    lora_state_dict = get_lora_state_dict(model)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': lora_state_dict, # Save only the LoRA weights
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'model_config': CFG.__dict__  # Save model configuration
    }
    
    torch.save(checkpoint, save_path)
    print(f"LoRA adapter saved to {save_path}")
    
    if is_best:
        print(f"New best LoRA adapter saved with loss: {loss:.4f}")


def train():
    """
    LoRA fine-tuning.
    """
    # Add argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning script for MiniChat model.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a pre-trained model checkpoint to load.')
    args = parser.parse_args()

    # 1. Setup Model and CFGuration
    model = MiniChat(CFG=CFG)
    
    # Load a pre-trained checkpoint if specified
    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        try:
            # Use map_location to ensure the model loads correctly regardless of the device it was saved on
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=CFG.device)["model_state_dict"])
            print("Checkpoint loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
            return
        except Exception as e:
            print(f"An error occurred while loading the checkpoint: {e}")
            return

    # 2. Freeze all original model parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 3. Apply LoRA to the model
    # This replaces the target linear layers with LoRALinear layers,
    # while preserving the original weights within the LoRALinear wrapper.
    apply_lora_to_model(model)
    model.to(CFG.device).to(torch.bfloat16).train()

    # 4. Unfreeze only the LoRA parameters for training
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True

    # Print trainable parameters to verify
    print("\nTrainable parameters:")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            print(name)
            trainable_params += param.numel()
    
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters (LoRA): {trainable_params}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%\n")


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
    train_data_loader = DataLoader(dataset, batch_size=2)

    # 6. Setup Optimizer and Loss Function
    # The optimizer will only update the parameters that have requires_grad=True
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    try:
        print("Checking if exist optimizer state_dict for continuing training...")
        optimizer.load_state_dict(torch.load(args.checkpoint_path, map_location=CFG.device)["optimizer_state_dict"])
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        print("Using default optimizer settings.")

    except Exception as e:
        print(f"An error occurred while loading the checkpoint: {e}")
        print("Using default optimizer settings.")

    criterion = nn.CrossEntropyLoss(ignore_index=200000)

    # 7. Training Loop
    model.train()
    num_epochs = 1
    device = CFG.device
    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        step = 0
        for batch in tqdm(train_data_loader):
            inputs, targets = batch
            inputs: torch.Tensor = inputs.to(device)
            targets: torch.Tensor = targets.to(device)
            # Find where assistant response starts (after the assistant marker)
            target_start_index = (inputs == 200014).nonzero(as_tuple=True)[1]
            optimizer.zero_grad()
            logits: torch.Tensor = model(inputs)  # Raw logits from model
            print(f"logits shape: {logits.shape}")
            
            # Properly mask targets: set non-assistant tokens to ignore_index
            for i in range(logits.shape[0]):
                # Mask everything before the assistant response (keep assistant marker, mask response tokens)
                # Assistant response starts AFTER the assistant marker
                mask_end = target_start_index[i] + 1  # +1 to include the assistant marker in masking
                targets[i, :mask_end] = 200000  # Use ignore_index, not 0
                

            logits = logits.contiguous().view(-1, CFG.vocab_size)
            print(f"logits shape: {logits.shape}")

            

            targets = targets.view(-1).long()
            print(f"targets shape: {targets.shape}")
            loss = criterion(logits, targets)
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

            
            epoch_loss += loss.item()
            batch_count += 1
            step += 1
            
            loss.backward()
            optimizer.step()

        # Log epoch-level metrics
        avg_epoch_loss = epoch_loss / batch_count

        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save latest model
        save_model(model, optimizer, epoch + 1, avg_epoch_loss, latest_model_path)
        
        # Save best model if this is the best loss so far
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_model(model, optimizer, epoch + 1, avg_epoch_loss, best_model_path, is_best=True)
            

    # Training completed - save final summary
    print(f"\nTraining completed!")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Latest model saved to: {latest_model_path}")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    train()
