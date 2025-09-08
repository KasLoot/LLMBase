from dataclasses import dataclass
import os
import torch

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    learning_rate: float = 1e-4
    epochs: int = 1
    batch_size: int = 1
    vision_model: str = "/home/yuxin/models/siglip2-large-patch16-256"
    language_model: str = "/home/yuxin/models/Qwen2.5-0.5B-Instruct"
    save_dir: str = "/home/yuxin/models/LLMBase/Qwen2.5_0.5B_Siglip2"
    processed_data_dir: str = "/home/yuxin/datasets/BUAADreamer/llava-en-zh-2k/qwen_tokenizer_processed_data/"
    seq_length: int = 2048
    train_type: str = "projector_only"  # "full" or "projector_only"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def text_tokens_path(self) -> str:
        return os.path.join(self.processed_data_dir, "file_0_texttokens.pt")
    
    @property
    def image_embeds_path(self) -> str:
        return os.path.join(self.processed_data_dir, "file_0_image_embeds.pt")
    
    @property
    def assistant_index_path(self) -> str:
        return os.path.join(self.processed_data_dir, "file_0_assistant_indexes.json")