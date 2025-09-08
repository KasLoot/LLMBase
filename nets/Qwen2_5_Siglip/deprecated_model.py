import json
from pathlib import Path
import os
import requests
import safetensors
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.clip.modeling_clip import CLIPVisionModel
from transformers.models.clip.image_processing_clip import CLIPImageProcessor



vision_backbone_name = "/home/yuxin/models/siglip2-large-patch16-256"
text_backbone_name = "/home/yuxin/models/Qwen2.5-0.5B-Instruct"

# _ = hf_hub_download(
#     vision_backbone_name, filename="pytorch_model.bin", local_dir="/content"
# )
# _ = hf_hub_download(
#     text_backbone_name, filename="model.safetensors", local_dir="/content"
# )

vision_config = AutoConfig.from_pretrained(vision_backbone_name).vision_config
text_config = AutoConfig.from_pretrained(text_backbone_name)

llava_config = LlavaConfig(vision_config=vision_config, text_config=text_config)

model = LlavaForConditionalGeneration(llava_config).cuda()


def load_weights(path_to_weights: str):
    if path_to_weights.endswith(".safetensors"):
        return load_safetensors_weights(path_to_weights)
    elif path_to_weights.endswith(".bin"):
        return load_bin_weights(path_to_weights)
    else:
        raise ValueError(f"Unsupported weights file: {path_to_weights}")

def load_bin_weights(path_to_weights: str):
    return torch.load(path_to_weights, weights_only=True)

def load_safetensors_weights(path_to_weights: str):
    return safetensors.torch.load_file(path_to_weights)


vision_backbone_state_dict = load_weights(os.path.join(vision_backbone_name, "model.safetensors"))
text_backbone_state_dict = load_weights(os.path.join(text_backbone_name, "model.safetensors"))

incompatible_keys = model.vision_tower.load_state_dict(
    vision_backbone_state_dict, strict=False
)

assert len(incompatible_keys.missing_keys) == 0, (
    f"Missing keys in state dict: {incompatible_keys.missing_keys}"
)

incompatible_keys = model.language_model.load_state_dict(
    text_backbone_state_dict, strict=False
)

_ = model.vision_tower.requires_grad_(False)
_ = model.language_model.requires_grad_(False)


# helper function to count model parameters
def count_parameters(model, trainable_only=False):
    return sum(
        p.numel()
        for p in model.parameters()
        if not trainable_only or p.requires_grad
    )

print(f"Total parameters: {count_parameters(model)}")
print(f"Trainable parameters: {count_parameters(model, trainable_only=True)}")


tokenizer = AutoTokenizer.from_pretrained(text_backbone_name)
print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}") # tokenizer.pad_token_id: 151643
# tokenizer.pad_token_id = 32001


LLAVA_CHAT_TEMPLATE = (
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "{% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"
)
tokenizer.chat_template = LLAVA_CHAT_TEMPLATE


sample_messages = [
    {
        "content": [
            {
                "index": 0,
                "text": None,
                "type": "image"
            },
            {
                "index": None,
                "text": "\nWhat potential activities might be popular at this location?",
                "type": "text"
            }
        ],
        "role": "user"
    },
    {
        "content": [
            {
                "index": None,
                "text": (
                    "At this location, with a sandy path leading to the ocean where multiple boats, including "
                    "sailboats, are moored, popular activities might include boating, sailing, swimming, and "
                    "beachcombing. Additionally, the sandy path and shoreline provide an ideal setting for leisurely "
                    "strolls and picnics, while the ocean view offers a serene environment for relaxation and "
                    "photography. Depending on the specific area and available facilities, other water sports such as "
                    "kayaking, paddleboarding, and snorkeling could also be prevalent."
                ),
                "type": "text"
            }
        ],
        "role": "assistant"
    }
]


tokenizer.apply_chat_template(
    sample_messages, tokenize=False, add_generation_prompt=False
)

tokenizer.image_token_id = 1805

processor = LlavaProcessor(
    image_processor=CLIPImageProcessor.from_pretrained(vision_backbone_name),
    tokenizer=tokenizer,
    patch_size=model.config.vision_config.patch_size,
)
processor.chat_template = LLAVA_CHAT_TEMPLATE

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)


train_dataset = load_dataset(
    "/home/yuxin/datasets/HuggingFaceH4/llava-instruct-mix-vsft", split="train", streaming=True
)

def get_data_collator(processor, ignore_index):
    def collate_examples(examples):
        # Extract texts and images from the raw examples
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"][0])
        print(f"Number of texts: {len(texts)}, Number of images: {len(images)}")
        print(f"images[0]: {images[0]}")
        print(f"texts[0]: {texts[0]}")
        # Process the inputs (tokenize text and transform images)
        batch = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=False)

        # Create labels
        labels = batch["input_ids"].clone()
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = ignore_index
        batch["labels"] = labels

        return batch

    return collate_examples







args = Seq2SeqTrainingArguments(
    output_dir="~/LLMBase/training_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=350,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 2e-5},
    warmup_ratio=0.05,
    logging_strategy="steps",
    logging_steps=5,
    fp16=True,
    remove_unused_columns=False,  # Important!
    optim="adamw_torch",
    report_to="none",
    save_strategy="no",  # let's not save the checkpoint to disk, otherwise it'll take 5 mins
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    data_collator=get_data_collator(
        processor, ignore_index=151643,
    ),
    train_dataset=train_dataset,
)


trainer.train()