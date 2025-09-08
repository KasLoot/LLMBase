"""
Multi-modal LLaVA-style model combining SigLIP vision encoder with Qwen2.5 language model.

This module implements a vision-language model that can process both images and text,
using SigLIP for vision encoding and Qwen2.5 for language understanding.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipVisionModel,
    SiglipImageProcessor,
)
from tqdm import tqdm
from train_config import TrainingConfig

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.dataset import Qwen_Siglip_SFT_Dataset



class VisionProjector(nn.Module):
    """
    A simple MLP to project vision embeddings to the LLM's embedding space.
    
    This module maps the vision encoder's output dimension to the language model's
    embedding dimension using a two-layer MLP with GELU activation.
    """
    
    def __init__(self, vision_hidden_dim: int, llm_hidden_dim: int):
        """
        Initialize the vision projector.
        
        Args:
            vision_hidden_dim: Dimension of vision encoder output
            llm_hidden_dim: Dimension of language model embeddings
        """
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project vision embeddings to LLM embedding space.
        
        Args:
            x: Vision embeddings tensor
            
        Returns:
            Projected embeddings compatible with LLM
        """
        return self.projector(x)


class LLaVAModel(nn.Module):
    """
    Multi-modal LLaVA-style model combining vision and language understanding.
    
    This model integrates a vision encoder (SigLIP) with a language model (Qwen2.5)
    through a learnable projector module. The vision and language models are frozen,
    and only the projector is trained.
    """
    
    def __init__(self, vision_tower: nn.Module, projector: nn.Module, language_model: nn.Module, train_type: str):
        """
        Initialize the LLaVA model.
        
        Args:
            vision_tower: Pre-trained vision encoder
            projector: Vision-to-language projector module
            language_model: Pre-trained language model
        """
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector
        self.language_model = language_model
        self.train_type = train_type
        assert train_type in ["full", "projector_only"], "train_type must be 'full' or 'projector_only'"

        if self.train_type == "full":
            print("Training type: full - Freezing Vision Tower only.")
            self._freeze_ViT_only()
        elif self.train_type == "projector_only":
            print("Training type: projector_only - Freezing both Vision Tower and Language Model.")
            self._freeze_base_models()



    def _freeze_base_models(self) -> None:
        """Freeze vision tower and language model parameters."""
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False

        # The projector is the only part we'll train initially
        for param in self.projector.parameters():
            param.requires_grad = True
    
    def _freeze_ViT_only(self) -> None:
        """Freeze only the vision tower parameters."""
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = True

        # The projector is the only part we'll train initially
        for param in self.projector.parameters():
            param.requires_grad = True

    def concat_image_text(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Concatenate image and text embeddings.
        
        Args:
            image_emb: Image embeddings (batch_size, img_seq_len, d_model)
            text_emb: Text embeddings (batch_size, text_seq_len, d_model)
            
        Returns:
            Combined embeddings with image embeddings inserted into text sequence
        """
        # Insert image embeddings into positions 1:257 of text embeddings
        text_emb[:, 1:257, :] = image_emb
        return text_emb

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the multi-modal model.
        
        Args:
            pixel_values: Pre-processed image embeddings
            input_ids: Tokenized text input
            attention_mask: Attention mask for text (currently unused)
            labels: Target labels for training loss calculation
            
        Returns:
            Model outputs including logits and optionally loss
        """
        # Project vision embeddings to LLM space
        projected_vision_emb = self.projector(pixel_values)

        # Get LLM input embeddings for text
        llm_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Combine image and text embeddings, final_embeds shape: torch.Size([1, 2048, 896])
        final_embeds = self.concat_image_text(projected_vision_emb, llm_embeds)

        # attention_mask shape: torch.Size([1, 2048, 2048])
        mask = torch.tril(torch.ones(final_embeds.shape[1], final_embeds.shape[1], device=final_embeds.device)).unsqueeze(0)

        # Pass combined embeddings through language model
        outputs = self.language_model(
            inputs_embeds=final_embeds,
            attention_mask=mask,  # Attention mask needs recalculation for new length
            labels=labels,  # For loss calculation during training
        )
        
        return outputs


def load_models(config: TrainingConfig) -> Tuple[nn.Module, nn.Module, object, object]:
    """
    Load and initialize the vision model, language model, and processors.
    
    Returns:
        Tuple of (vision_tower, language_model, image_processor, tokenizer)
    """
    print("Loading models...")
    
    # Load vision model (SigLIP)
    vision_tower = SiglipVisionModel.from_pretrained(config.vision_model).to(config.device)
    image_processor = SiglipImageProcessor.from_pretrained(config.vision_model)

    # Load language model (Qwen2.5) with float16 precision
    language_model = AutoModelForCausalLM.from_pretrained(
        config.language_model,
        torch_dtype=torch.float16,  # Important for Qwen2
        device_map={"": 0}  # Automatically place on GPU 0
    )
    tokenizer = AutoTokenizer.from_pretrained(config.language_model, trust_remote_code=True)

    print("Models loaded successfully!")
    return vision_tower, language_model, image_processor, tokenizer


def create_projector(vision_dim: int, llm_dim: int, config: TrainingConfig) -> VisionProjector:
    """
    Create and initialize the vision projector.
    
    Args:
        vision_dim: Vision encoder output dimension
        llm_dim: Language model embedding dimension
        
    Returns:
        Initialized projector module
    """
    projector = VisionProjector(vision_dim, llm_dim).to(config.device).to(torch.bfloat16)
    print(f"Projector created: Maps from {vision_dim} (SigLIP) to {llm_dim} (Qwen2.5)")
    return projector


def print_model_info(model: nn.Module) -> None:
    """Print model summary and parameter counts."""
    print(f"LLaVA model:\n{model}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


def create_dataset_and_dataloader(config: TrainingConfig) -> DataLoader:
    """
    Create dataset and dataloader for training.
    
    Args:
        config: Training configuration
        
    Returns:
        DataLoader for training
    """
    dataset = Qwen_Siglip_SFT_Dataset(
        text_tokens_path=config.text_tokens_path,
        image_embeds_path=config.image_embeds_path,
        assistant_index_path=config.assistant_index_path,
        seq_length=config.seq_length
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    return dataloader

def save_model(model, optimizer, epoch, loss, save_path, is_best=False):
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def train_model(model: LLaVAModel, dataloader: DataLoader, config: TrainingConfig) -> None:
    """
    Train the multi-modal model.
    
    Args:
        model: The LLaVA model to train
        dataloader: Training data loader
        config: Training configuration
    """
    # Optimizer only updates projector parameters
    optimizer = torch.optim.Adam(model.projector.parameters(), lr=config.learning_rate)
    
    print("Starting training...")

    for epoch in tqdm(range(config.epochs)):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids, image_embeds, targets = batch
            input_ids = input_ids.to(config.device)
            image_embeds = image_embeds.to(config.device)
            targets = targets.to(config.device).long()

            # Forward pass
            outputs = model(
                pixel_values=image_embeds,
                input_ids=input_ids,
                labels=targets
            )
            
            # Calculate loss and backpropagate
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            # Clear cache to prevent memory issues
            torch.cuda.empty_cache()
            
            # For demo purposes, only train on first batch
            break
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        save_model(model, optimizer, epoch + 1, avg_loss, os.path.join(config.save_dir, f"Qwen2.5_0.5B_Siglip2_epoch{epoch+1}.pt"))

    print("Training finished.")


def main():
    """Main function to set up and train the model."""

    # Set up training
    config = TrainingConfig()

    # Load models and processors
    vision_tower, language_model, image_processor, tokenizer = load_models(config)

    # Get model dimensions
    vision_dim = vision_tower.config.hidden_size
    llm_dim = language_model.config.hidden_size
    
    # Create projector
    projector = create_projector(vision_dim, llm_dim, config=config)

    
    # Create complete model, train_type="full" or "projector_only"
    model = LLaVAModel(vision_tower, projector, language_model, train_type=config.train_type).to(torch.bfloat16).to(config.device)

    # Print model information
    print_model_info(model)
    
    
    dataloader = create_dataset_and_dataloader(config)
    
    # Train the model
    train_model(model, dataloader, config)


if __name__ == "__main__":
    main()
