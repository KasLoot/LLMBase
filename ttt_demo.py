from transformers import AutoTokenizer
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

# Initializing a TTT ttt-1b style configuration
# configuration = TTTConfig(**TTT_STANDARD_CONFIGS['1b']) is equivalent to the following
configuration = TTTConfig()

# Initializing a model from the ttt-1b style configuration
model = TTTForCausalLM(configuration)
model.eval()

# Accessing the model configuration
configuration = model.config

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/yuxin/Documents/models/Llama-2-7b-chat-hf')

# Prefill
input_ids = tokenizer("Greeting from TTT!", return_tensors="pt").input_ids
logits = model(input_ids=input_ids)
print(logits)

# Decoding
out_ids = model.generate(input_ids=input_ids, max_length=50)
out_str = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
print(out_str)