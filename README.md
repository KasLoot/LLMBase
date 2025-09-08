# LLMBase Use Guide
## Models
- Qwen2.5_Siglip2: LLaVa architecture with Qwen2.5 and Siglip2
- MiniChat-1B: Transformer decoder-based text model
- MiniChat_Vision: LLaVa architecture (Unified Embedding) with MiniChat-1B and Siglip2
- MiniChat_MoE: Mixture of Experts text model
- gpt_oss: the open source gpt_oss model


## Data Handling
All training scripts expect data to be pre-processed and read from `.pt` files. You may want to convert your data into this format, or write a custom Dataset class to read from your data format.

### Example: Preparing data for Qwen2.5_Siglip2
- BUAADreamer/llava-en-zh-2k/en/
```bash
cd process_data/sft

# pre-process data for MiniChat_Vision (modify the path in the script to your data path)
python BUAADreamer.py

# pre-process data for Qwen2.5_Siglip2 (modify the path in the script to your data path)
python BUAADreamer_qwen.py
```


## Training

Clone the repo and sync dependencies:
```bash
git clone https://github.com/KasLoot/LLMBase.git
cd LLMBase
uv sync
```

### Qwen2.5_Siglip2

```bash
cd nets/Qwen2_5_Siglip
```
Run training:
- training configs can be adjusted in `train_config.py`
```bash
python model_formatted.py
```


### MiniChat-1B

```bash
cd LLMBase
```

#### Pre-training
```bash
# Single GPU training
python pretrain.py

# Distributed training
python pretrain_ddp.py
```
#### SFT
```bash
# Single GPU training --Full parameter fine-tuning
python sft.py

# Single GPU training --LoRA fine-tuning
python sft_lora.py

# Distributed training --Full parameter fine-tuning (bugs to be fixed)
python sft_ddp.py
```

### MiniChat_Vision

```bash
cd LLMBase
```
#### Training
```bash
# Single GPU training
python train_miniChat_Vision.py
```