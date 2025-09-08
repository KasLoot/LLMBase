import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipVisionModel,
    SiglipImageProcessor,
    BitsAndBytesConfig
)
from PIL import Image

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load the Vision Model (The Eyes) ---
# We use SiglipVisionModel as we only need the encoder part.
vision_tower_id = "/home/yuxin/models/siglip2-large-patch16-256"
vision_tower = SiglipVisionModel.from_pretrained(vision_tower_id).to(device)
image_processor = SiglipImageProcessor.from_pretrained(vision_tower_id)

# --- Load the Language Model (The Brain) ---
# We use BitsAndBytes for 4-bit quantization to save memory.
llm_id = "/home/yuxin/models/Qwen2.5-0.5B-Instruct"

language_model = AutoModelForCausalLM.from_pretrained(
    llm_id,
    torch_dtype=torch.float16, # Important for Qwen2
    device_map={"": 0} # Automatically place on GPU 0
)
tokenizer = AutoTokenizer.from_pretrained(llm_id, trust_remote_code=True)

# print(f"language model:\n{language_model}")

print("Models loaded successfully!")



class VisionProjector(nn.Module):
    """A simple MLP to project vision embeddings to the LLM's embedding space."""
    def __init__(self, vision_hidden_dim, llm_hidden_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim)
        )

    def forward(self, x):
        return self.projector(x)
    

# Get dimensions
vision_dim = vision_tower.config.hidden_size
llm_dim = language_model.config.hidden_size

# Instantiate the projector
projector = VisionProjector(vision_dim, llm_dim).to(device).to(torch.bfloat16)

print(f"Projector created: Maps from {vision_dim} (SigLIP) to {llm_dim} (Qwen2).")



# # --- Add a special token for the image placeholder ---
# # It's good practice to use a token that the tokenizer is unlikely to have seen.
# IMAGE_TOKEN = "<image>"
# tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
# # Resize the token embeddings to account for the new token
# language_model.resize_token_embeddings(len(tokenizer))
# language_model.get_output_embeddings().weight.data = language_model.get_input_embeddings().weight.data
# IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

class LLaVAModel(nn.Module):
    def __init__(self, vision_tower, projector: nn.Module, language_model, train_type):
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector
        self.language_model = language_model
        # train_type: "full" or "projector_only"
        self.train_type = train_type

        

        if self.train_type == "full":
            # Activate projector and language model parameters for training, keep ViT frozen
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            for param in self.language_model.parameters():
                param.requires_grad = True
            for param in self.projector.parameters():
                param.requires_grad = True

        elif self.train_type == "projector_only":
            # Freeze the ViT and Language models
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            for param in self.language_model.parameters():
                param.requires_grad = False
            # Activate the projector for training
            for param in self.projector.parameters():
                param.requires_grad = True


    def concat_image_text(self, image_emb: torch.Tensor, text_emb: torch.Tensor):
        # image_emb: (batch_size, img_seq_len, d_model)
        # text_emb: (batch_size, text_seq_len, d_model)

        text_emb[:, 1:257, :] = image_emb

        return text_emb
    

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        pixel_values = self.projector(pixel_values)


        llm_embeds = self.language_model.get_input_embeddings()(input_ids)

        final_embeds = self.concat_image_text(pixel_values, llm_embeds)

        mask = torch.tril(torch.ones(final_embeds.shape[1], final_embeds.shape[1], device=final_embeds.device)).unsqueeze(0)

        outputs = self.language_model(
            inputs_embeds=final_embeds,
            attention_mask=None, # Attention mask needs to be recalculated for the new length
            labels=labels, # Optional: for calculating loss during training
        )
        return outputs

# Instantiate the final model
model = LLaVAModel(vision_tower, projector, language_model, train_type="full").to(torch.bfloat16).to(device="cuda")

# Print model summary
print(f"LLaVa model:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_trainable_params}")


# --- Conceptual Training Loop ---
# This is a simplified example. A real implementation would require a proper dataset and dataloader.

# Optimizer will only update the projector's parameters
optimizer = torch.optim.Adam(model.projector.parameters(), lr=1e-4)
num_epochs = 1 # In reality, you'd train for longer


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from torch.utils.data import DataLoader
from utils.dataset import Qwen_Siglip_SFT_Dataset
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class TrainingConfig_local:
    learning_rate: float = 1e-4
    epochs: int = 1
    batch_size: int = 1
    model_file: str = "/home/yuxin/models/LLMBase/MiniChat/MiniChat-1B-pt-v1.pt"
    save_dir: str = "checkpoints"
    processed_data_dir: str = "/home/yuxin/datasets/BUAADreamer/llava-en-zh-2k/qwen_tokenizer_processed_data/"
    text_tokens_path: str = os.path.join(processed_data_dir, "file_0_texttokens.pt")
    image_embeds_path: str = os.path.join(processed_data_dir, "file_0_image_embeds.pt")
    assistant_index_path: str = os.path.join(processed_data_dir, "file_0_assistant_indexes.json")
    seq_length: int = 2048

train_CFG = TrainingConfig_local

dataset = Qwen_Siglip_SFT_Dataset(
text_tokens_path=train_CFG.text_tokens_path,
image_embeds_path=train_CFG.image_embeds_path,
assistant_index_path=train_CFG.assistant_index_path,
seq_length=train_CFG.seq_length
)

dataloader = DataLoader(dataset, batch_size=train_CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)




print("Starting conceptual training step...")
for epoch in range(num_epochs):
    optimizer.zero_grad()

    batch_loss = 0
    for b, batch in tqdm(enumerate(dataloader)):
        
        input_ids, image_embeds, targets = batch
        input_ids = input_ids.to("cuda")
        image_embeds = image_embeds.to("cuda")
        targets = targets.to("cuda").long()

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
        batch_loss += loss.item()
        print(f"Epoch {epoch+1}, batch {b+1} Loss: {loss.item()}")
        torch.cuda.empty_cache()
        break
    print(f"Epoch {epoch+1}, Average Loss: {batch_loss / train_CFG.batch_size}")

print("\nConceptual training finished. The projector has been updated.")