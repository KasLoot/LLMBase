from utils.dataset import VLM_SFT_Dataset
import torch
from torch.utils.data import DataLoader


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length assistant_indexes"""
    text_tokens = torch.stack([item[0] for item in batch])
    image_embeds = torch.stack([item[1] for item in batch])
    assistant_indexes = [item[2] for item in batch]  # Keep as list of lists
    return text_tokens, image_embeds, assistant_indexes



dataset = VLM_SFT_Dataset(
    text_tokens_path="/workspace/datasets/BUAADreamer/llava-en-zh-2k/processed_data/file_0_texttokens.pt",
    image_embeds_path="/workspace/datasets/BUAADreamer/llava-en-zh-2k/processed_data/file_0_image_embeds.pt",
    assistant_index_path="/workspace/datasets/BUAADreamer/llava-en-zh-2k/processed_data/file_0_assistant_indexes.json",
    seq_length=2048
)

inputs, image_embeds, targets = dataset[0]
print(inputs.shape)
print(image_embeds.shape)
print(targets.shape)
print(targets)


# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_fn)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)


for batch in dataloader:
    inputs, image_embeds, targets = batch
    print("Batch:")
    print(inputs)
    print(image_embeds.shape)
    print(targets)
    print(inputs)

    break

