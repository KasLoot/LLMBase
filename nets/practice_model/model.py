import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        """
        Initializes the Rotary Positional Embeddings module.

        Args:
            dim (int): The feature dimension of the input vectors (e.g., query, key).
                       Must be an even number.
            max_seq_len (int): The maximum sequence length the model will see.
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError("Dimension must be an even number.")
            
        # 1. Pre-compute the theta values
        # theta_i = 1 / (10000^(2i/d)) for i in [0, 1, ..., d/2 - 1]
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        print(f"Theta shape: {theta.shape}")  # Debugging line
        print(f"Theta values: {theta}")  # Debugging line
        
        # 2. Pre-compute the position indices (m) and the frequencies (m * theta)
        # shape: (max_seq_len, dim/2)
        m = torch.arange(max_seq_len)
        print(f"Position indices shape: {m.shape}")  # Debugging line
        print(f"Position indices values: {m}")  # Debugging line

        freqs = torch.outer(m, theta) # Computes m * theta for all pairs
        print(f"Frequencies shape: {freqs.shape}")  # Debugging line
        print(f"Frequencies values: {freqs}")  # Debugging line
        
        # 3. Convert frequencies to complex numbers for rotation e^(j * m * theta)
        # shape: (max_seq_len, dim/2)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        print(f"Frequencies (complex) shape: {freqs_complex.shape}")  # Debugging line
        print(f"Frequencies (complex) values: {freqs_complex}")  # Debug
        
        # Register freqs_complex as a buffer. It's part of the model's state
        # but not a parameter that requires gradients. It will be moved to the
        # correct device (e.g., GPU) along with the model.
        self.register_buffer("freqs_complex", freqs_complex)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
                              or any other shape with the last dimension being the feature dim.

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input.
        """
        # Get the sequence length from the input tensor
        # x.shape = (batch, seq_len, n_heads, head_dim)
        seq_len = x.shape[1]

        x = x.float().reshape(*x.shape[:-1], -1, 2)
        print(f"Input reshaped to complex shape: {x.shape}")  # Debugging line
        # print(f"Input values (reshaped): {x}")  # Debugging line
        
        # 1. Reshape x to view the last dimension as pairs for complex numbers
        # x_complex shape: (batch, seq_len, n_heads, head_dim/2)
        x_complex = torch.view_as_complex(x)
        
        # 2. Get the pre-computed frequencies for the current sequence length
        # freqs_complex shape: (seq_len, head_dim/2)
        freqs_for_seq = self.freqs_complex[:seq_len]
        
        # Reshape freqs for broadcasting with the input tensor's shape.
        # freqs_reshaped shape: (1, seq_len, 1, head_dim/2)
        freqs_reshaped = freqs_for_seq.unsqueeze(0).unsqueeze(2)
        
        # 3. Apply the rotation by element-wise multiplication in the complex plane
        # This is the core operation: x * e^(j*theta)
        # x_rotated shape: (batch, seq_len, n_heads, head_dim/2)
        x_rotated = x_complex * freqs_reshaped
        
        # 4. Convert the complex numbers back to the real-valued representation
        # x_out shape: (batch, seq_len, n_heads, head_dim/2, 2)
        x_out = torch.view_as_real(x_rotated)
        
        # 5. Reshape back to the original input shape
        # x_out shape: (batch, seq_len, n_heads, head_dim)
        x_out = x_out.reshape(*x.shape)
        
        return x_out.type_as(x)

# --- Example Usage ---
batch_size = 2
seq_len = 4
n_heads = 2
head_dim = 32 # Must be even

# Create dummy Query and Key tensors
q = torch.randn(batch_size, seq_len, n_heads, head_dim)
k = torch.randn(batch_size, seq_len, n_heads, head_dim)

# Initialize RoPE module
rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=1024)

# Apply RoPE
q_rotated = rope(q)
k_rotated = rope(k)

print("Original Q shape:", q.shape)
print("Rotated Q shape:", q_rotated.shape)