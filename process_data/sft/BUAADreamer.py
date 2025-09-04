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
from transformers import AutoModel, AutoProcessor

import PIL
from PIL import Image


def get_image_encoder():
    ckpt = "/home/yuxin/models/siglip2-large-patch16-256"
    model = AutoModel.from_pretrained(ckpt, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(ckpt)

    return model, processor


class BUAADreamer():
    def __init__(self, local_dir, dataset_names: list[str], sq_l, text_tokenizer: tiktoken.Encoding, vit_model, vit_processor):
        self.local_dir = local_dir
        self.dataset_names = dataset_names
        self.sq_l = sq_l
        self.text_tokenizer = text_tokenizer
        self.vit_model = vit_model
        self.vit_processor = vit_processor

    def handle_datasets(self):
        for dataset_name in self.dataset_names:
            print(f"Processing dataset: {dataset_name}")
            if dataset_name == "llava-en-zh-2k":
                self.process_llava_en_zh_2k()
            else:
                print(f"Dataset {dataset_name} not recognized or not implemented.")

    def length_checker(self, data):
        pass



    def process_llava_en_zh_2k(self):
        dataset_path = os.path.join(self.local_dir, "llava-en-zh-2k/en/")
        print(f"Processing dataset at: {dataset_path}")
        pathlist = Path(dataset_path).rglob('*.parquet')
        max_text_len = 2048 - 258

        for f_i, file in enumerate(pathlist):
            print(f"Processing file: {file}")
            data = load_dataset("parquet", data_files=str(file))["train"]
            print(data[0])

            image_reserved_tokens_tensor = torch.tensor([[200009] + [200011]*256 + [200010]]*len(data), dtype=torch.int32)
            file_text_tensor = torch.ones((len(data), self.sq_l-258), dtype=torch.int32)*200000
            file_text_tensor = torch.hstack((image_reserved_tokens_tensor, file_text_tensor))
            print(f"File text tensor shape: {file_text_tensor}")

            file_image_embeds_tensor = torch.ones((len(data), 256, 1024), dtype=torch.bfloat16)

            # this stores the list of assistant indexes for each sample in the file [[index of <assistant> in sample 1], [index of <assistant> in sample 2], ...]
            file_assistant_indexes = []
            for i, sample in enumerate(tqdm(data)):
                torch.cuda.empty_cache()

                image = self.vit_processor(images=sample["images"][0], return_tensors="pt").to("cuda")
                image_embeds: torch.Tensor = self.vit_model.vision_model(**image).last_hidden_state
                file_image_embeds_tensor[i] = image_embeds.to(torch.bfloat16).detach().cpu()
                
                
                system_text_token = []
                if "system" in sample and sample["system"].strip() != "":
                    system_text_token = [200001] + self.text_tokenizer.encode(sample["system"]) + [200002]
                    if len(system_text_token) > max_text_len:
                        file_text_tensor[i, 0] = 220000  # Truncation token
                        print(f"sample {i} system message too long, truncated indicator set.")
                        break
                
                conversation_token = []
                assistant_indexes = []
                for msg in sample["messages"]:
                    if msg["role"] == "user":
                        role_token = [200003] + self.text_tokenizer.encode(msg["content"]) + [200004]
                    elif msg["role"] == "assistant":
                        role_token = [200005] + self.text_tokenizer.encode(msg["content"]) + [200006]
                    else:
                        print(f"Unknown role: {msg['role']}")
                        exit()
                    
                    if len(system_text_token) + len(role_token) > max_text_len:
                        file_text_tensor[i, 0] = 220000
                        print(f"sample {i} conversation too long, truncated indicator set.")
                    else:
                        if msg["role"] == "assistant":
                            assistant_index = len(conversation_token) + len(system_text_token) + 258
                            assistant_indexes.append(assistant_index)
                        conversation_token.extend(role_token)
                file_assistant_indexes.append(assistant_indexes)
                full_text_token = system_text_token + conversation_token
                assert len(full_text_token) <= max_text_len, f"Text length exceeds max limit: {len(full_text_token)} > {max_text_len}"
                file_text_tensor[i, 258:258+len(full_text_token)] = torch.tensor(full_text_token, dtype=torch.int32)

                # print(f"assistant tokens: {file_text_tensor[0][file_assistant_indexes[0]]}")

            # Filter out samples with truncation indicator (220000 at index 0)
            valid_mask = file_text_tensor[:, 0] != 220000
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # Keep only valid samples
                file_text_tensor_filtered = file_text_tensor[valid_indices]
                file_image_embeds_tensor_filtered = file_image_embeds_tensor[valid_indices]
                file_assistant_indexes_filtered = [file_assistant_indexes[i] for i in valid_indices.tolist()]
                
                print(f"Filtered out {len(data) - len(valid_indices)} samples due to truncation. Keeping {len(valid_indices)} valid samples.")
                
                torch.save(file_text_tensor_filtered, f"/home/yuxin/datasets/BUAADreamer/llava-en-zh-2k/BUAADreamer_processed/file_{f_i}_texttokens.pt")
                torch.save(file_image_embeds_tensor_filtered, f"/home/yuxin/datasets/BUAADreamer/llava-en-zh-2k/BUAADreamer_processed/file_{f_i}_image_embeds.pt")
                json.dump(file_assistant_indexes_filtered, open(f"/home/yuxin/datasets/BUAADreamer/llava-en-zh-2k/BUAADreamer_processed/file_{f_i}_assistant_indexes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            else:
                print(f"No valid samples found in file {f_i}. All samples were truncated.")
            break



            # file_json = []

            # for i, sample in enumerate(data):
            #     sample_json = {
            #         "messages": sample["messages"],
            #         "images": []
            #     }

            #     for img_idx, image in enumerate(sample["images"]):
            #         image_path = os.path.join(self.local_dir, f"llava-en-zh-2k/BUAADreamer_processed/images/image_{f_i}_{i}_{img_idx}.png")
            #         image.save(image_path)
            #         sample_json["images"].append(image_path)

            #     file_json.append(sample_json)

            # output_dir = os.path.join(self.local_dir, "llava-en-zh-2k/BUAADreamer_processed")
            # os.makedirs(output_dir, exist_ok=True)
            # output_file = os.path.join(output_dir, f"file_{f_i}.json")
            # json.dump(file_json, open(output_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            # break

vit_model, vit_processor = get_image_encoder()

allenai = BUAADreamer(local_dir="/home/yuxin/datasets/BUAADreamer/", dataset_names=["llava-en-zh-2k"], sq_l=2048, text_tokenizer=get_tokenizer(), vit_model=vit_model, vit_processor=vit_processor)
allenai.process_llava_en_zh_2k()