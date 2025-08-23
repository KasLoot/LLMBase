import torch
from collections import OrderedDict

def analyze_checkpoint_config(checkpoint_path):
    """
    Analyze a model checkpoint to determine the configuration parameters
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    state_dict = checkpoint["model_state_dict"]
    
    config = {}
    
    # 1. Extract vocab_size from embedding layer
    embedding_weight = state_dict['embedding.emb.weight']
    config['vocab_size'] = embedding_weight.shape[0]
    config['d_model'] = embedding_weight.shape[1]
    
    # 2. Extract number of transformer blocks
    transformer_blocks = 0
    for key in state_dict.keys():
        if key.startswith('transformer.transformer.') and key.endswith('.rms1.weight'):
            block_num = int(key.split('.')[2])
            transformer_blocks = max(transformer_blocks, block_num + 1)
    config['transformer_blocks'] = transformer_blocks
    
    # 3. Extract num_heads from attention weight shapes
    # The wq, wk, wv weights should have shape [d_model, d_model]
    wq_weight = state_dict['transformer.transformer.0.mhAttention.wq.weight']
    assert wq_weight.shape[0] == wq_weight.shape[1] == config['d_model']
    
    # 4. Extract hidden_size from feed-forward layer
    ffw1_weight = state_dict['transformer.transformer.0.ffw1.0.weight']
    config['hidden_size'] = ffw1_weight.shape[0]
    
    # 5. Extract sequence length from RoPE frequencies
    rope_cos = state_dict['transformer.transformer.0.mhAttention.rope_cos']
    config['sq_l'] = rope_cos.shape[0]
    
    # 6. Calculate head dimension and verify num_heads
    # d_k = d_model / num_heads, and rope frequencies should have shape [seq_len, d_k//2]
    rope_freq_dim = rope_cos.shape[1]
    d_k = rope_freq_dim * 2  # RoPE uses half dimensions
    config['num_h'] = config['d_model'] // d_k
    
    # 7. Other standard parameters (these are architectural choices, not inferred from weights)
    config['drop_rate'] = 0.1  # This can't be determined from saved weights
    
    # 8. Determine device and dtype
    sample_tensor = embedding_weight
    config['device'] = str(sample_tensor.device)
    config['dtype'] = str(sample_tensor.dtype)
    
    return config

def compare_configs(inferred_config, original_config_class):
    """
    Compare inferred config with the original config class defaults
    """
    print("=== CONFIGURATION ANALYSIS ===\n")
    
    print("Inferred Configuration from Checkpoint:")
    print("-" * 40)
    for key, value in inferred_config.items():
        print(f"{key:20}: {value}")
    
    print("\nOriginal ModelConfig Defaults:")
    print("-" * 40)
    original_config = original_config_class()
    for attr in ['sq_l', 'd_model', 'num_h', 'hidden_size', 'drop_rate', 'vocab_size', 'transformer_blocks']:
        if hasattr(original_config, attr):
            value = getattr(original_config, attr)
            print(f"{attr:20}: {value}")
    
    print("\nDifferences:")
    print("-" * 40)
    differences_found = False
    for attr in ['sq_l', 'd_model', 'num_h', 'hidden_size', 'vocab_size', 'transformer_blocks']:
        if hasattr(original_config, attr) and attr in inferred_config:
            original_val = getattr(original_config, attr)
            inferred_val = inferred_config[attr]
            if original_val != inferred_val:
                print(f"{attr:20}: {original_val} → {inferred_val}")
                differences_found = True
    
    if not differences_found:
        print("No differences found in comparable parameters!")
    
    return inferred_config

if __name__ == "__main__":
    from nets.MiniChat.model import ModelConfig
    
    checkpoint_path = "/Users/liuyuxin/Documents/models/MiniChat/MiniChat-1B-pt.pt"
    
    try:
        inferred_config = analyze_checkpoint_config(checkpoint_path)
        compare_configs(inferred_config, ModelConfig)
        
        print("\n=== CORRECTED CONFIGURATION ===")
        print("Use this configuration for inference:")
        print(f"""
@dataclass
class ModelConfig:
    sq_l: int = {inferred_config['sq_l']}
    d_model: int = {inferred_config['d_model']}
    num_h: int = {inferred_config['num_h']}
    hidden_size: int = {inferred_config['hidden_size']}
    drop_rate: float = 0.1  # Standard value
    vocab_size: int = {inferred_config['vocab_size']}
    transformer_blocks: int = {inferred_config['transformer_blocks']}
    device: torch.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
""")
        
    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")
        import traceback
        traceback.print_exc()
