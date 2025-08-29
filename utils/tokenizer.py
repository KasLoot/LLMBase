import tiktoken

def get_tokenizer():
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|reserved_199999|>": 199999,
            "<|pad|>": 200000,
            "<|system|>": 200001,
            "<|/system|>": 200002,
            "<|user|>": 200003,
            "<|/user|>": 200004,
            "<|assistant|>": 200005,
            "<|/assistant|>": 200006,
            "<|function|>": 200007,
            "<|/function|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|reserved_200012|>": 200012,
            "<|reserved_200013|>": 200013,
            "<|reserved_200014|>": 200014,
            "<|reserved_200015|>": 200015
        } | {
            f"<|reserved_{i}|>": i for i in range(200016, 201088)
        },
    )
    return tokenizer
