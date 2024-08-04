# PyTorch for implementing LLM (No GPU)
import torch

# Neural network modules and functions from PyTorch
from torch import nn
from torch.nn import functional as F

# NumPy for numerical operations
import numpy as np

# Matplotlib for plotting Loss etc.
from matplotlib import pyplot as plt

# Time module for tracking execution time
import time

# Pandas for data manipulation and analysis
import pandas as pd

# urllib for handling URL requests (Downloading Dataset)
import urllib.request

# Configuration object for model parameters
MASTER_CONFIG = {
    # Adding parameters later
}

# Read the content of the dataset
lines = open("tinyshakespeare.txt", 'r').read()

# Create a sorted list of unique characters in the dataset
vocab = sorted(list(set(lines)))

# Display the first 10 characters in the vocabulary list
print('Printing the first 10 characters of the vocab list:', vocab[:10])

# Output the total number of characters in our dataset (Vocabulary Size)
print('Total number of characters in our dataset (Vocabulary Size):', len(vocab))

# Mapping integers to characters (itos)
itos = {i: ch for i, ch in enumerate(vocab)}

# Mapping characters to integers (stoi)
stoi = {ch: i for i, ch in enumerate(vocab)}

# Encode function: Converts a string to a list of integers using the mapping stoi
def encode(s):
    return [stoi[ch] for ch in s]

# Decode function: Converts a list of integers back to a string using the mapping itos
def decode(l):
    return ''.join([itos[i] for i in l])

# Example: Encode the string "hello" and then decode the result
print(encode("hello, good morning"))

# Convert the dataset into a torch tensor with specified data type (dtype)
dataset = torch.tensor(encode(lines), dtype=torch.int8)

# Display the shape of the resulting tensor
print(dataset.shape)

# # Function to get batches for training, validation, or testing
# def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
#     # Split the dataset into training, validation, and test sets
#     train = data[:int(.8 * len(data))]
#     val = data[int(.8 * len(data)): int(.9 * len(data))]
#     test = data[int(.9 * len(data)):]

#     # Determine which split to use
#     batch_data = train
#     if split == 'val':
#         batch_data = val
#     if split == 'test':
#         batch_data = test

#     # Pick random starting points within the data
#     ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))

#     # Create input sequences (x) and corresponding target sequences (y)
#     x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
#     y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

#     return x, y


# # Update the MASTER_CONFIG with batch_size and context_window parameters
# MASTER_CONFIG.update({
#     'vocab_size': len(vocab),  # Number of unique characters in the dataset
#     'batch_size': 8,          # Number of batches to be processed at each random split
#     'context_window': 16      # Number of characters in each input (x) and target (y) sequence of each batch
# })

# # Obtain batches for training using the specified batch size and context window
# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# # Decode the sequences to obtain the corresponding text representations
# decoded_samples = [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

# # Print the random sample
# print(decoded_samples)

# @torch.no_grad()  # Don't compute gradients for this function
# def evaluate_loss(model, config=MASTER_CONFIG):
#     # Placeholder for the evaluation results
#     out = {}
    
#     # Set the model to evaluation mode
#     model.eval()

#     # Iterate through training and validation splits
#     for split in ["train", "val"]:
#         # Placeholder for individual losses
#         losses = []

#         # Generate 10 batches for evaluation
#         for _ in range(10):
#             # Get input sequences (xb) and target sequences (yb)
#             xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            
#             # Perform model inference and calculate the loss
#             _, loss = model(xb, yb)
            
#             # Append the loss to the list
#             losses.append(loss.item())

#         # Calculate the mean loss for the split and store it in the output dictionary
#         out[split] = np.mean(losses)
    
#     # Set the model back to training mode
#     model.train()
    
#     return out

# def get_rotary_matrix(context_window, embedding_dim):
#     # Initialize a tensor for the rotary matrix with zeros
#     R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    
#     # Loop through each position in the context window
#     for position in range(context_window):
#         # Loop through each dimension in the embedding
#         for i in range(embedding_dim // 2):
#             # Calculate the rotation angle (theta) based on the position and embedding dimension
#             theta = 10000. ** (-2. * (i - 1) / embedding_dim)
#             # Calculate the rotated matrix elements using sine and cosine functions
#             m_theta = position * theta
#             R[position, 2 * i, 2 * i] = np.cos(m_theta)
#             R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
#             R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
#             R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
#     return R


# class RMSNorm(nn.Module):
#     def __init__(self, layer_shape, eps=1e-8, bias=False):
#         super(RMSNorm, self).__init__()

#         # Registering a learnable parameter 'scale' as a parameter of the module
#         self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

#     def forward(self, x):
#         """
#         Assumes shape is (batch, seq_len, d_model)
#         """
#         # Calculating the Frobenius norm, RMS = 1/sqrt(N) * Frobenius norm
#         ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5

#         # Normalizing the input tensor 'x' with respect to RMS
#         raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)

#         # Scaling the normalized tensor using the learnable parameter 'scale'
#         return self.scale[:x.shape[1], :].unsqueeze(0) * raw


# # Define the SimpleModel_RMS with RMSNorm
# class RoPEAttentionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         # Linear transformation for query
#         self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
#         # Linear transformation for key
#         self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
#         # Linear transformation for value
#         self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
#         # Obtain rotary matrix for positional embeddings
#         self.R = get_rotary_matrix(config['context_window'], config['d_model'])

#     def get_rotary_matrix(context_window, embedding_dim):
#         # Initialize a tensor for the rotary matrix with zeros
#         R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        
#         # Loop through each position in the context window
#         for position in range(context_window):
#             # Loop through each dimension in the embedding
#             for i in range(embedding_dim // 2):
#                 # Calculate the rotation angle (theta) based on the position and embedding dimension
#                 theta = 10000. ** (-2. * (i - 1) / embedding_dim)
#                 # Calculate the rotated matrix elements using sine and cosine functions
#                 m_theta = position * theta
#                 R[position, 2 * i, 2 * i] = np.cos(m_theta)
#                 R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
#                 R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
#                 R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
#         return R


#     def forward(self, x, return_attn_weights=False):
#         # x: input tensor of shape (batch, sequence length, dimension)

#         b, m, d = x.shape  # batch size, sequence length, dimension

#         # Linear transformations for Q, K, and V
#         q = self.w_q(x)
#         k = self.w_k(x)
#         v = self.w_v(x)

#         # Rotate Q and K using the RoPE matrix
#         q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m].to('cuda'))).transpose(0, 1)
#         k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m].to('cuda'))).transpose(0, 1)

#         # Perform scaled dot-product attention
#         activations = F.scaled_dot_product_attention(
#             q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
#         )

#         if return_attn_weights:
#             # Create a causal attention mask
#             attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
#             # Calculate attention weights and add causal mask
#             attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
#             attn_weights = F.softmax(attn_weights, dim=-1)
#             return activations, attn_weights

#         return activations


# class RoPEMaskedMultiheadAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         # Create a list of RoPEMaskedAttentionHead instances as attention heads
#         self.heads = nn.ModuleList([
#             RoPEAttentionHead(config).to('cuda') for _ in range(config['n_heads'])
#         ])
#         self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])  # Linear layer after concatenating heads
#         self.dropout = nn.Dropout(.1)  # Dropout layer

#     def forward(self, x):
#         # x: input tensor of shape (batch, sequence length, dimension)

#         # Process each attention head and concatenate the results
#         heads = [h(x) for h in self.heads]
#         x = torch.cat(heads, dim=-1)
        
#         # Apply linear transformation to the concatenated output
#         x = self.linear(x)
        
#         # Apply dropout
#         x = self.dropout(x)
#         return x


# MASTER_CONFIG.update({
#     'd_model': 128,
# })

# MASTER_CONFIG.update({
#     'n_heads': 8,
# })


# class RopeModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         # Embedding layer for input tokens
#         self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        
#         # RMSNorm layer for pre-normalization
#         self.rms = RMSNorm((config['context_window'], config['d_model']))
        
#         # RoPEMaskedMultiheadAttention layer
#         self.rope_attention = RoPEMaskedMultiheadAttention(config)

#         # Linear layer followed by ReLU activation
#         self.linear = nn.Sequential(
#             nn.Linear(config['d_model'], config['d_model']),
#             nn.ReLU(),
#         )

#         # Final linear layer for prediction
#         self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

#         print("model params:", sum([m.numel() for m in self.parameters()]))

#     def forward(self, idx, targets=None):
#         idx = idx.to('cuda')
#         # idx: input indices
#         x = self.embedding(idx)

#         # One block of attention
#         x = self.rms(x)  # RMS pre-normalization
#         x = x + self.rope_attention(x)

#         x = self.rms(x)  # RMS pre-normalization
#         x = x + self.linear(x)

#         logits = self.last_linear(x)

#         if targets is not None:
#             targets = targets.to('cuda')
#             loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
#             return logits, loss

#         else:
#             return logits



# # Update MASTER_CONFIG with training parameters
# MASTER_CONFIG.update({
#     'epochs': 1000,          # Number of training epochs
#     'log_interval': 10,      # Log information every 10 batches during training
#     'batch_size': 32,        # Increase batch size to 32
# })



# # Function to perform training
# def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False, device='cuda'):
#     model.train()
#     # Placeholder for storing losses
#     losses = []
    
#     # Start tracking time
#     start_time = time.time()

#     # Iterate through epochs
#     for epoch in range(config['epochs']):
#         # Zero out gradients
#         optimizer.zero_grad()

#         # Obtain batches for training
#         xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])
#         xs.to('cuda')
#         ys.to('cuda')
#         # Forward pass through the model to calculate logits and loss
#         logits, loss = model(xs, targets=ys)

#         # Backward pass and optimization step
#         loss.backward()
#         optimizer.step()

#         # If a learning rate scheduler is provided, adjust the learning rate
#         if scheduler:
#             scheduler.step()

#         # Log progress every specified interval
#         if epoch % config['log_interval'] == 0:
#             # Calculate batch time
#             batch_time = time.time() - start_time
            
#             # Evaluate loss on validation set
#             x = evaluate_loss(model)
            
#             # Store the validation loss
#             losses += [x]
            
#             # Print progress logs if specified
#             if print_logs:
#                 print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
                
#             # Reset the timer
#             start_time = time.time()

#             # Print learning rate if a scheduler is provided
#             if scheduler:
#                 print("lr: ", scheduler.get_lr())

#     # Print the final validation loss
#     print("Validation loss: ", losses[-1]['val'])
    
#     # Plot the training and validation loss curves
#     return losses

# # Instantiate the SimpleBrokenModel with updated configuration
# model = RopeModel(MASTER_CONFIG).to('cuda')
# # Print the total number of parameters in the model
# print("Total number of parameters in the Simple Neural Network Model:", sum([m.numel() for m in model.parameters()]))

# # Obtain batches for training using the specified batch size and context window
# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
# print(xs.shape, ys.shape)

# # Define the Adam optimizer for model parameters
# optimizer = torch.optim.Adam(
#     model.parameters(),      # Pass the model parameters to the optimizer
# )
# # Execute the training process
# loss = train(model, optimizer, print_logs=False)
# df = pd.DataFrame(loss)

# # Plotting the data
# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df['train'], label='Train Loss', marker='o')
# plt.plot(df.index, df['val'], label='Validation Loss', marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss over Epochs')
# plt.legend()
# plt.grid(True)
# plt.xticks(df.index, labels=[f'Epoch {i+1}' for i in df.index])

# # Display the plot
# plt.show()

# # Generate function for text generation using the trained model
# def generate(model, config=MASTER_CONFIG, max_new_tokens=30):
#     idx = torch.zeros(5, 1).long().to('cuda')
#     for _ in range(max_new_tokens):
#         # Call the model
#         logits = model(idx[:, -config['context_window']:])
#         last_time_step_logits = logits[
#             :, -1, :
#         ]  # all the batches (1), last time step, all the logits
#         p = F.softmax(last_time_step_logits, dim=-1)  # softmax to get probabilities
#         idx_next = torch.multinomial(
#             p, num_samples=1
#         )  # sample from the distribution to get the next token
#         idx = torch.cat([idx, idx_next], dim=-1)  # append to the sequence
#     return [decode(x) for x in idx.tolist()]

# # Generate text using the trained model
# generation = generate(model.to('cuda'))
# print(generation)

