from sys import path
import torch
from torch.utils.data import Dataset
import os
import json

    

class LanguageModelDataset(Dataset):
    """Custom PyTorch Dataset for next-token prediction."""
    def __init__(self, path, seq_length):
        """
        Args:
            path (str): Path to the tokenized data file
            seq_length (int): Length of each sequence
        """
        self.seq_length = seq_length
        # Load the flat tensor of tokens
        tokens = torch.load(path)
        
        # Split the flat tensor into chunks of seq_length + 1 
        # (extra token for target prediction)
        chunk_size = seq_length + 1
        num_chunks = len(tokens) // chunk_size
        
        # Reshape into chunks, discarding incomplete final chunk
        self.samples = tokens[:num_chunks * chunk_size].view(num_chunks, chunk_size)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index and splits it into
        input and target tensors.
        """
        # Get the full sequence for the given index
        sequence = self.samples[idx]

        # Input is all tokens except the last one
        input_ids = sequence[:-1]
        # Target is all tokens except the first one (shifted by 1)
        target_ids = sequence[1:]

        return input_ids, target_ids
    


class LanguageModel_SFT_Dataset(Dataset):
    """Custom PyTorch Dataset for next-token prediction."""
    def __init__(self, dir: str, seq_length: int):
        """
        Args:
            paths (list[str]): List of paths to the tokenized data files
            seq_length (int): Length of each sequence
        """
        self.seq_length = seq_length

        self.tokens = []
        for file in os.listdir(dir):
            chunk = torch.load(os.path.join(dir, file))
            self.tokens.append(chunk)
            print(f"Loaded {file} with {len(chunk)} tokens.")
        self.tokens = torch.cat(self.tokens)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.tokens)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index and splits it into
        input and target tensors.
        """
        # Get the full sequence for the given index
        sample: torch.Tensor = self.tokens[idx]
        sep_index = torch.where(sample == 200005)
        print(f"Separator index: {sep_index}")

        input = sample
        target = sample.clone()[sep_index + 1:]

        return sep_index, input, target
    

class VLM_SFT_Dataset(Dataset):
    def __init__(self, text_tokens_path, image_embeds_path, assistant_index_path, seq_length):
        self.seq_length = seq_length
        self.text_tokens = torch.load(text_tokens_path)
        print(f"self.text_tokens shape: {self.text_tokens.shape}")
        self.image_embeds = torch.load(image_embeds_path)
        self.assistant_indices = json.load(open(assistant_index_path, 'r'))
        assert len(self.text_tokens) == len(self.image_embeds) == len(self.assistant_indices), "All inputs must have the same number of samples."
        assert self.text_tokens.shape[1] == self.seq_length+1, f"Text tokens must have sequence length {self.seq_length}."
        print(f"Loaded dataset with {len(self.text_tokens)} samples.")

    def __len__(self):
        return len(self.text_tokens)
    
    def __getitem__(self, idx):
        input_token: torch.Tensor = self.text_tokens[idx][:-1]
        image_embed: torch.Tensor = self.image_embeds[idx]
        last_assistant_index: list = self.assistant_indices[idx][-1]

        target_token = self.text_tokens[idx][1:]
        target_token[:last_assistant_index+1] = 200000

        return input_token, image_embed, target_token