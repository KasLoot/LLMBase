import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from cfg.cfg import MASTER_CONFIG
from torch.utils.data import Dataset
import tiktoken
import json
import random
import os
from transformers.tokenization_utils_base import PaddingStrategy

class VQADataset(Dataset):
    def __init__(self, path, tokenizer):
        self.path = path
        self.enc = tiktoken.get_encoding('cl100k_base')
        # Encode the dataset into a torch tensor with specified data type (dtype)
        self.dataset = json.load(open(self.path))
        self.encoded_dataset = []
        
        # for block in self.dataset:
        #     self.encoded_dataset.append([torch.tensor(self.encode(block['system'] + block['task'] + block['query']), dtype=torch.int32), torch.tensor(self.encode(block['response']),dtype=torch.int32)])
                  
    def preprocess(text, tokenizer):
        return tokenizer.encode(text, truncation=True, padding='max_length')    
    # Encode function: Converts a string to a list of integers using the mapping stoi
    def encode(self, s):
        return self.enc.encode(s)

    # Decode function: Converts a list of integers back to a string using the mapping itos
    def decode(self, l):
        return self.enc.decode(l)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.encoded_dataset[idx][0], self.encoded_dataset[idx][1]
    
    # def split_data(self, ratio=[8,2,0]):
    #     assert sum(ratio) == 10, "Ratios must sum to 10"
    #     random.shuffle(self.encoded_dataset)
    #     train = self.encoded_dataset [:int(len(self.dataset ) * ratio[0] / 10)]
    #     val = self.encoded_dataset [int(len(self.dataset ) * ratio[0] / 10): int(len(self.dataset ) * (ratio[0] + ratio[1]) / 10)]
    #     test = self.encoded_dataset [int(len(self.dataset ) * (ratio[0] + ratio[1]) / 10):]
    #     return train, val, test
