import torch
from torch.utils.data import Dataset, DataLoader
from nets.demo_model.model import ModelConfig
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


def process_data(tokenizer, CFG: ModelConfig):

    # assert tokenizer == None, "Missing tokenizer"

    tiny_textbook = load_dataset("parquet", data_files="/home/yuxin/datasets/tiny-textbooks/tiny-textbooks/train-00000-of-00001.parquet")

    raw_text: list[str]= tiny_textbook["train"]["textbook"]

    # longest sample token len = 3214



    max_sq = CFG.sq_l
    dataset_tokens = torch.full((len(raw_text), max_sq), 200000)
    start_tensor = torch.full((len(raw_text), 1), 199998)
    dataset_tokens = torch.hstack((start_tensor, dataset_tokens))

    idx = 0
    for sample in tqdm(raw_text):


        tk = tokenizer.encode(sample, allowed_special="all") + [199999]

        if len(tk) > max_sq + 1:
            tk = tk[:max_sq + 1]
            dataset_tokens[idx, 1:1+len(tk)] = torch.tensor(tk)
        else:
            dataset_tokens[idx, 1:1+len(tk)] = torch.tensor(tk)
        
        idx += 1

    torch.save(dataset_tokens, "/home/yuxin/datasets/LLMBase/tiny_textbook.pt")
    print("dataset tensors saved to: /home/yuxin/datasets/LLMBase/tiny_textbook.pt")

    # test = torch.load("/home/yuxin/datasets/LLMBase/tiny_textbook.pt")
    
    # assert torch.equal(test, dataset_tokens)



class LanguageModelDataset(Dataset):
    """Custom PyTorch Dataset for next-token prediction."""
    def __init__(self, path):
        """
        Args:
            tokenized_text (torch.Tensor): A tensor of tokenized text
                                           of shape (num_samples, sequence_length).
        """
        self.samples = torch.load(path)


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

        # The input is the sequence up to the last token
        input_ids = sequence[:-1]

        # The target is the sequence from the second token onwards
        target_ids = sequence[1:]

        return input_ids, target_ids