import torch

from tqdm import tqdm

from utils.tokenizer import get_tokenizer
from nets.MiniChat.model import ModelConfig
from datasets import load_dataset

from dataset_handler import handle_dataset


CFG = ModelConfig()

    



def pad_token(tokenizer, token) -> torch.Tensor:
    max_sq = CFG.sq_l
    end_index = 0

    padded_tokens = torch.full((len(token), max_sq), 200000)
    start_tensor = torch.full((len(token), 1), 199998)
    padded_tokens = torch.hstack((start_tensor, padded_tokens))



    idx = 0
    for sample in tqdm(token):

        tk = tokenizer.encode(sample, allowed_special="all") + [199999]
        if len(tk) > max_sq + 1:
            tk = tk[:max_sq + 1]
            padded_tokens[idx, 1:] = torch.tensor(tk)[:max_sq] # shrink tokens
            end_index = max_sq
        else:
            padded_tokens[idx, 1:1+len(tk)] = torch.tensor(tk) # copy tokens to padded tokens
            end_index = 1+len(tk)
        
        idx += 1

    return end_index, padded_tokens


def preprocess_pretrain_data_deprecate():
    tokenizer = get_tokenizer()
    print("tokenizer initialized")
    # assert tokenizer == None, "Missing tokenizer"

    tiny_textbook = load_dataset("parquet", data_files="/home/yuxin/datasets/tiny-textbooks/tiny-textbooks/train-00000-of-00001.parquet")

    raw_text: list[str]= tiny_textbook["train"]["textbook"]
    # raw_text shape: (399000)

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
            dataset_tokens[idx, 1:] = torch.tensor(tk)[:max_sq]
        else:
            dataset_tokens[idx, 1:1+len(tk)] = torch.tensor(tk)
        
        idx += 1

    torch.save(dataset_tokens, "/home/yuxin/datasets/LLMBase/train_tiny_textbook.pt")
    print("dataset tensors saved to: /home/yuxin/datasets/LLMBase/train_tiny_textbook.pt")

    # test = torch.load("/home/yuxin/datasets/LLMBase/tiny_textbook.pt")
    
    # assert torch.equal(test, dataset_tokens)

def preprocess_pretrain_data(datasets :dict):
    for dataset in datasets:
        raw_data = handle_dataset(dataset)
        


        pass


def preprocess_sft_data(data :list[dict]):
    tokenizer = get_tokenizer()
    print("tokenizer initialized")
    max_sq = CFG.sq_l
    dataset_tokens = torch.full((len(data), max_sq), 200000)
    start_tensor = torch.full((len(data), 1), 199998)
    dataset_tokens = torch.hstack((start_tensor, dataset_tokens))

    

    idx = 0
    for sample in tqdm(data):



        tk = [200013] + tokenizer.encode(sample["system"], allowed_special="all") \
            + [200015] + tokenizer.encode(sample["assistant"], allowed_special="all") \
            + [200014] + tokenizer.encode(sample["user"], allowed_special="all") + [199999]

        if len(tk) > max_sq + 1:
            tk = tk[:max_sq + 1]
            dataset_tokens[idx, 1:] = torch.tensor(tk)[:max_sq]
        else:
            dataset_tokens[idx, 1:1+len(tk)] = torch.tensor(tk)
        
        idx += 1

    SAVE_PATH = "/Users/liuyuxin/Documents/LLMBase/data_samples/sft_data.pt"
    torch.save(dataset_tokens, SAVE_PATH)
    print("dataset tensors saved to: {SAVE_PATH}")

    # test = torch.load("/home/yuxin/datasets/LLMBase/tiny_textbook.pt")
    
    # assert torch.equal(test, dataset_tokens)








test_data = [{
            "system": "You are a helpful assistant.",
            "user": "What is the capital of France?",
            "assistant": "The capital of France is Paris."
        },
        {
            "system": "You are a helpful assistant.",
            "user": "What is the capital of Germany?",
            "assistant": "The capital of Germany is Berlin."
        }
    ]

# preprocess_sft_data(test_data)