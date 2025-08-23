import os
import torch
from datasets import load_dataset



def handle_dataset(dataset):
    match dataset["name"]:
        case "tiny_textbook":
            raw_text = []
            for dataset_path in dataset["paths"]:
                raw_text.extend(load_dataset("parquet", data_files=dataset_path)["train"]["textbook"])
            return raw_text
        case "another_dataset":
            # Handle another_dataset
            return None
        case _:
            # Handle unknown dataset
            return None