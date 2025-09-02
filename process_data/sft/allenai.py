import torch
import os
import json
import tiktoken
from tqdm import tqdm
import sys
from datasets import load_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.tokenizer import get_tokenizer
from pathlib import Path

class Allenai():
    def __init__(self, local_dir, dataset_names: list[str], sq_l, tokenizer: tiktoken.Encoding):
        self.local_dir = local_dir
        self.dataset_names = dataset_names
        self.sq_l = sq_l
        self.tokenizer = tokenizer

    def handle_datasets(self):
        for dataset_name in self.dataset_names:
            print(f"Processing dataset: {dataset_name}")
            if dataset_name == "tulu-v2-sft-mixture":
                self.process_tulu_v2_sft_mixture()
            else:
                print(f"Dataset {dataset_name} not recognized or not implemented.")

    def lenth_checker(self, data):
        pass


    def process_tulu_v2_sft_mixture_deprecated(self):
        dataset_path = os.path.join(self.local_dir, "tulu-v2-sft-mixture/data/")
        for file in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file)
            if file.endswith(".parquet"):
                print(f"Processing file: {file_path}")
                data = load_dataset("parquet", data_files=file_path)['train']["messages"]
                print(len(data))
                start_of_text = torch.full((len(data), 1), 199998)
                dataset = torch.full((len(data), self.sq_l-1), 200000)
                dataset = torch.hstack((start_of_text, dataset))
                false_index = []
                for i in tqdm(range(len(data)),desc=f"Processing samples"):
                    for j in range(len(data[i])):
                        sample_temp = []
                        token = []
                        if data[i][j]['role'] == 'user':
                            token = [200003] + self.tokenizer.encode(data[i][j]['content'], allowed_special="all") + [200004]
                            if len(sample_temp) + len(token) > self.sq_l - 1:
                                false_index.append(i)
                                break
                            sample_temp.extend(token)
                        elif data[i][j]['role'] == 'assistant':
                            token = [200005] + self.tokenizer.encode(data[i][j]['content'], allowed_special="all") + [200006]
                            if len(sample_temp) + len(token) > self.sq_l - 1:
                                print("assistance message exceed max sequence length, truncating overflow.")
                                sample_temp.extend(token[:self.sq_l - 1 - len(sample_temp)])
                                break
                        elif data[i][j]['role'] == 'system':
                            token = [200001] + self.tokenizer.encode(data[i][j]['content'], allowed_special="all") + [200002]
                            if len(sample_temp) + len(token) > self.sq_l - 1:
                                false_index.append(i)
                                break
                            sample_temp.extend(token)
                        else:
                            print(f"undefined role {data[i][j]['role']}, exiting")
                            exit(0)
                    
                    dataset[i, 1:len(sample_temp)+1] = torch.tensor(sample_temp)
                
                print(f"Original dataset size: {len(dataset)}")
                print(f"False data indices count: {len(false_index)}")
                keep_indices = [i for i in tqdm(range(len(dataset)), desc="Filtering false data") if i not in set(false_index)]
                dataset = dataset[keep_indices]
                print(f"Filtered dataset size: {len(dataset)}")

                TENSOR_SAVE_DIR = os.path.join(self.local_dir, "tulu-v2-sft-mixture/token_tensors")
                # if directory exists
                if not os.path.exists(TENSOR_SAVE_DIR):
                    os.makedirs(TENSOR_SAVE_DIR)
                torch.save(dataset, os.path.join(TENSOR_SAVE_DIR, f"{file}_token_tensors.pt"))
                # filtered_data = train_dataset.filter(lambda example: example['dataset'] == 'flan_v2')
                # print(filtered_data)
                # # This will print the first example from the filtered dataset
                # if len(filtered_data) > 0:
                #     print(filtered_data[0])

        return 0
    def process_tulu_v2_sft_mixture(self):
        dataset_path = os.path.join(self.local_dir, "tulu-v2-sft-mixture/data/")
        pathlist = Path(dataset_path).rglob('*.parquet')
        for file in pathlist:
            print(f"Processing file: {file}")
            data = load_dataset("parquet", data_files=file)

        return 0
    

    def process_WildChat(self):
        dataset_path = os.path.join(self.local_dir, "WildChat-4.8M/data/")
        print(f"Processing dataset at: {dataset_path}")
        pathlist = Path(dataset_path).rglob('*.parquet')


        for f_i, file in enumerate(pathlist):
            print(f"Processing file: {file}")
            data = load_dataset("parquet", data_files=str(file))["train"]["conversation"]
            print(data[0][0]["content"])
            print(data[0][1]["content"])

            file_json = []

            for s_i, sample in enumerate(data):
                sample_json = {
                    "conversations": [],
                    "tools": None
                }

                for m_i, msg in enumerate(sample):
                    role = "system" if msg["role"] == "system" else ("user" if m_i % 2 == 0 else "assistant")
                    sample_json["conversations"].append(msg["role"])

                



            break

allenai = Allenai(local_dir="/home/yuxin/datasets/allenai/", dataset_names=["tulu-v2-sft-mixture"], sq_l=2048, tokenizer=get_tokenizer())
allenai.process_WildChat()