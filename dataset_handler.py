import os
from tqdm import tqdm
from datasets import load_dataset



def handle_dataset(datasets):
    print(f"Processing datasets from author: {datasets['author']}")
    raw_text_all_in_one = ""
    if datasets["author"] == "nampdn-ai":
        for dataset in tqdm(datasets["datasets"], desc="Processing nampdn-ai datasets"):
            if dataset["name"] == "tiny_textbook":
                    for dataset_file in tqdm(os.listdir(dataset["dir"]), desc="Processing files in tiny_textbook"):
                        if dataset_file.endswith(".parquet"):
                            file_data = load_dataset("parquet", data_files=os.path.join(dataset["dir"], dataset_file))["train"]["textbook"]
                            merged_string = "\n\n".join(file_data)
                            raw_text_all_in_one += merged_string
    elif datasets["author"] == "open-phi":
        for dataset in tqdm(datasets["datasets"], desc="Processing open-phi datasets"):
            if dataset["name"] == "textbooks_grounded":
                for dataset_file in tqdm(os.listdir(dataset["dir"]), desc="Processing files in textbooks_grounded"):
                    if dataset_file.endswith(".parquet"):
                        file_data = load_dataset("parquet", data_files=os.path.join(dataset["dir"], dataset_file))
                        topic_column = file_data["train"]["topic"]
                        content_column = file_data["train"]["markdown"]
                        combined_entries = [f"{topic}\n{content}" for topic, content in zip(topic_column, content_column)]
                        merged_string = "\n\n".join(combined_entries)
                        raw_text_all_in_one += merged_string
    print("Finished processing dataset.")
    return raw_text_all_in_one
