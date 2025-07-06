import torch
import torch.nn.functional as F

from nets.net import DemoModel
from nets.llama3 import Llama3
from nets.llama3_d import Llama3_D
from utils.tokenizer import Tokenizer

CONFIG = {
    "batch_size": 1,
    "max_sequence": 512,
    "emb_dim": 512,  
    "num_head": 8,
    "dropout_rate": 0.1,
    "enc_num_block": 6,
    "decoder_num_block": 6,
    "input_mask": None,
    "output_mask": None,
    "vocab_size": 200019
}

Llama3_CONFIG = {
    "batch_size": 1,
    "max_sequence": 512,
    "emb_dim": 512,  
    "num_head": 8,
    "dropout_rate": 0.1,
    "num_block": 6,
    "input_mask": None,
    "output_mask": None,
    "vocab_size": 200019,
}

Llama3_D_CONFIG = {
    "batch_size": 2,
    "max_sequence": 128,
    "out_max_sequence": 256,
    "emb_dim": 512,  
    "num_head": 16,
    "dropout_rate": 0.1,
    "num_block": 32,
    "input_mask": None,
    "output_mask": None,
    "vocab_size": 200019,
}

print(Llama3_D_CONFIG)

torch.manual_seed(10)
torch.set_default_dtype(torch.bfloat16)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

sentences = ["User: What is the capital of France? Assistant: The capital of France is Paris.", "User: Can you tell me about the Eiffel Tower? Assistant: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."]
tokenizer = Tokenizer(model_name='o200k_base', max_sequence=Llama3_D_CONFIG["max_sequence"])
# print(tokenizer.encode("<|pad|>"))
# print(tokenizer.decode([199999]))
print(f"Special token set: {tokenizer.special_tokens_set}")
print(f"Vocab size: {tokenizer.vocab_size}")
tokenized_sentences = [tokenizer.encode(sentence) for sentence in sentences]
# print(f"Tokenized sentences: {tokenized_sentences}\n")



# batch_size, max_sequence, emb_dim, num_head, dropout_rate, enc_num_block, decoder_num_block, input_mask, output_mask, vocab_size = (
#     CONFIG["batch_size"],
#     CONFIG["max_sequence"],
#     CONFIG["emb_dim"],
#     CONFIG["num_head"],
#     CONFIG["dropout_rate"],
#     CONFIG["enc_num_block"],
#     CONFIG["decoder_num_block"],
#     CONFIG["input_mask"],
#     CONFIG["output_mask"],
#     CONFIG["vocab_size"]
# )
# model = Llama3(Llama3_CONFIG).to(device)
# print(f"Model: {model}\n")
memory = torch.cuda.memory_allocated()
model = Llama3_D(Llama3_D_CONFIG).to(device)
model_memory = torch.cuda.memory_allocated() - memory
print(f"Model memory usage: {model_memory / 1e6} Mb")
print(model)

# Forward Pass
memory = torch.cuda.memory_allocated()
outputs: torch.Tensor = model(torch.tensor(tokenized_sentences).to(device))
print(f"Llama_D model: output size: {outputs.size()}")
outputs = F.softmax(outputs, dim=-1)
outputs = torch.argmax(outputs, dim=-1).tolist()  # Get the index of the maximum value in the last dimension
forward_pass_memory = torch.cuda.max_memory_allocated() - memory
print(f"Forward pass memory usage: {forward_pass_memory / 1e6} Mb")

i = 0
for batch in outputs:
    print(f"Batch {i+1} outputs:")
    for output in batch:
        # print(output)
        try:
            print(tokenizer.decode([output]), end="")
        except Exception as e:
            print(f"<|Invalid Token|>", end="")
    print("\n")
    i += 1























# torch.cuda.empty_cache()  # Clear GPU memory before running the model

# B, S, E = CONFIG["batch_size"], CONFIG["max_sequence"], CONFIG["emb_dim"]
# memory_before = torch.cuda.memory_allocated()
# input_embeddings = torch.randn((B, S, E)).to(device)
# tgt_embeddings = torch.randn((B, S, E)).to(device)
# print(f"GPU memory for input and target: {(torch.cuda.memory_allocated() - memory_before)/1e6} Mb")

# print(f"Input size: {input_embeddings.size()}\n")

# memory_before = torch.cuda.memory_allocated()
# with torch.no_grad():
#     attention = Transformer_Block(CONFIG["emb_dim"], CONFIG["num_head"], CONFIG["max_sequence"], CONFIG["dropout_rate"], CONFIG["enc_num_block"], CONFIG["decoder_num_block"], CONFIG["input_mask"], CONFIG["output_mask"], CONFIG["vocab_size"]).to(device)
#     print(f"GPU memory for model: {(torch.cuda.memory_allocated() - memory_before)/1e6} Mb")
#     output = attention(input_embeddings, tgt_embeddings)
#     output = F.softmax(output, dim=-1)  # Apply softmax to the output
#     print(f"GPU memory for forward pass: {(torch.cuda.max_memory_allocated() - memory_before)/1e6} Mb")


# print(f"Output size: {output.size()}  {output.dtype}\n{output[0][0][:8]}")