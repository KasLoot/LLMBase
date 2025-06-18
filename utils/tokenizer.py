import tiktoken


class Tokenizer(tiktoken.Encoding):
    def __init__(self, model_name='o200k_base', max_sequence=None):
        assert max_sequence is not None, "max_sequence must be specified"
        self.max_sequence = max_sequence
        o200k_base = tiktoken.get_encoding(model_name)
        self.tokenizer = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
            name="o200k_base",
            pat_str=o200k_base._pat_str,
            mergeable_ranks=o200k_base._mergeable_ranks,
            special_tokens={
                **o200k_base._special_tokens,
                "<|pad|>": 100264
            }
        )
        self.vocab_size = self.tokenizer.n_vocab
        self.special_tokens_set = self.tokenizer.special_tokens_set
        
    def padding(self, token_ids):
        """Pads the token IDs to the maximum sequence length."""
        if len(token_ids) < self.max_sequence:
            token_ids += [100264] * (self.max_sequence - len(token_ids))
        return token_ids[:self.max_sequence]

    def encode(self, text):
        """Encodes a string into a list of token IDs."""
        encoded = self.tokenizer.encode(text + "<|endoftext|>", allowed_special="all")
        encoded = encoded[:self.max_sequence]  # Limit to max_sequence
        encoded = self.padding(encoded)
        return encoded

    def decode(self, token_ids):
        """Decodes a list of token IDs back into a string."""
        return self.tokenizer.decode(token_ids)
    
    