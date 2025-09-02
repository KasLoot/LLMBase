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

import PIL
from PIL import Image


class BUAADreamer():
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


  
    def process_BUAADreamer(self):
        dataset_path = os.path.join(self.local_dir, "llava-en-zh-2k/en/")
        print(f"Processing dataset at: {dataset_path}")
        pathlist = Path(dataset_path).rglob('*.parquet')

        for f_i, file in enumerate(pathlist):
            print(f"Processing file: {file}")
            data = load_dataset("parquet", data_files=str(file))["train"]
            print(data[0])

            file_json = []

            for i, sample in enumerate(data):
                sample_json = {
                    "messages": sample["messages"],
                    "images": []
                }

                for img_idx, image in enumerate(sample["images"]):
                    image_path = os.path.join(self.local_dir, f"llava-en-zh-2k/BUAADreamer_processed/images/image_{f_i}_{i}_{img_idx}.png")
                    image.save(image_path)
                    sample_json["images"].append(image_path)

                file_json.append(sample_json)

            output_dir = os.path.join(self.local_dir, "llava-en-zh-2k/BUAADreamer_processed")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"file_{f_i}.json")
            json.dump(file_json, open(output_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            break

allenai = BUAADreamer(local_dir="/home/yuxin/datasets/BUAADreamer/", dataset_names=["tulu-v2-sft-mixture"], sq_l=2048, tokenizer=get_tokenizer())
allenai.process_BUAADreamer()