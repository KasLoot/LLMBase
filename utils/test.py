# --- preprocess.py ---
import os
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import ByteLevelBPETokenizer

# --- 1. Load Raw Data ---
print("Loading raw parquet files...")
raw_datasets = load_dataset('parquet', data_files='path/to/your/data/*.parquet')

# --- 2. Get a Tokenizer ---
# Option A: Train a new one (if pretraining from scratch)
# tokenizer = ByteLevelBPETokenizer()
# def get_training_corpus(dataset):
#     for i in range(0, len(dataset["train"]), 1000):
#         yield dataset["train"][i : i + 1000]["text"]
# tokenizer.train_from_iterator(get_training_corpus(raw_datasets), vocab_size=50257, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
# tokenizer.save_model(".", "my-awesome-tokenizer")
# tokenizer_path = "my-awesome-tokenizer/vocab.json" # Path to the trained file

# Option B: Or use a pre-existing one (more common)
tokenizer_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
# --- 3. Define Processing Functions ---
block_size = 1024 # Your model's context length

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"])

# Chunking function
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # For decoders, labels are the same as the input_ids
    result["labels"] = result["input_ids"].copy()
    return result

# --- 4. Process and Save the Dataset ---
print("Tokenizing and chunking dataset...")

# Note: The 'map' function automatically uses caching. The results are stored in Arrow files.
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=os.cpu_count(), # Use all available CPU cores
    remove_columns=["text"]
)

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=os.cpu_count(),
)

# The magic step: save the processed dataset to disk
save_path = "processed_lm_dataset"
print(f"Saving processed dataset to {save_path}...")
lm_datasets.save_to_disk(save_path)

print("✅ Pre-processing complete.")