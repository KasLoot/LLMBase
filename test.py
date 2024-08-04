import tiktoken

enc = tiktoken.get_encoding('cl100k_base')
text = "Hello, world!<|endoftext|>"
encoded_text = enc.encode(text, allowed_special="all")
print(encoded_text)

