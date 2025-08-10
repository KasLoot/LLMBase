# Rotary Position Embedding (RoPE) Implementation

This document describes the implementation of Rotary Position Embedding (RoPE) in the DemoTextModel.

## Overview

Rotary Position Embedding (RoPE) is a method for encoding positional information in transformer models that has several advantages over traditional positional encodings:

1. **Relative Position Awareness**: RoPE naturally encodes relative positions between tokens
2. **Length Extrapolation**: Models can handle sequences longer than those seen during training
3. **No Parameter Overhead**: RoPE doesn't add learnable parameters to the model

## Implementation Details

### Core Functions

#### `precompute_rope_freqs(seq_len, d_k, base=10000, device="cpu")`

Precomputes the cosine and sine frequencies used for rotation.

**Parameters:**
- `seq_len`: Maximum sequence length
- `d_k`: Head dimension (must be even)
- `base`: Base frequency for computation (default: 10000)
- `device`: Device to place tensors on

**Returns:**
- `freqs_cos`: Cosine frequencies of shape `(seq_len, d_k // 2)`
- `freqs_sin`: Sine frequencies of shape `(seq_len, d_k // 2)`

#### `rotary_positional_embedding(x, freqs_cos, freqs_sin)`

Applies rotary position embedding to input tensor.

**Parameters:**
- `x`: Input tensor of shape `(batch_size, num_heads, seq_len, head_dim)`
- `freqs_cos`: Precomputed cosine frequencies
- `freqs_sin`: Precomputed sine frequencies

**Returns:**
- Tensor with RoPE applied, same shape as input

### Mathematical Foundation

RoPE works by rotating the query and key vectors in the attention mechanism. For a vector `x = [x1, x2, x3, x4, ...]`, RoPE applies:

```
x_rotated = [
    x1 * cos(θ) - x2 * sin(θ),
    x1 * sin(θ) + x2 * cos(θ),
    x3 * cos(θ/base) - x4 * sin(θ/base),
    x3 * sin(θ/base) + x4 * cos(θ/base),
    ...
]
```

where `θ` depends on the position in the sequence and the frequency.

### Integration with Attention

RoPE is integrated into the `MHAttention` class:

1. **Initialization**: Frequencies are precomputed once during model creation
2. **Forward Pass**: RoPE is applied to query and key vectors before attention computation
3. **Device Handling**: Frequencies are registered as buffers to automatically move with the model

## Key Properties

### 1. Magnitude Preservation
RoPE preserves the magnitude of vectors:
```python
||RoPE(x)|| = ||x||
```

### 2. Rotation Invertibility
Applying RoPE with opposite angles returns to the original vector:
```python
RoPE(RoPE(x, θ), -θ) = x
```

### 3. Relative Position Encoding
The dot product between rotated vectors depends only on their relative positions:
```python
RoPE(q_i) · RoPE(k_j) = f(q_i, k_j, i-j)
```

## Usage

The RoPE implementation is automatically used when creating a `DemoTextModel`:

```python
from nets.demo_model.model import DemoTextModel, ModelConfig

config = ModelConfig(
    sq_l=1024,      # sequence length
    d_model=512,    # model dimension
    num_h=8,        # attention heads
    # ... other config
)

model = DemoTextModel(config)
```

## Performance Considerations

1. **Memory**: RoPE frequencies are precomputed and cached, using `O(seq_len * d_k)` memory
2. **Computation**: RoPE adds minimal computational overhead compared to standard attention
3. **Device Placement**: Frequencies are automatically moved to the correct device with the model

## Testing

The implementation includes comprehensive tests:

- `test_rope.py`: Basic functionality tests
- `test_rope_detailed.py`: Mathematical property verification
- `test_training_rope.py`: Training integration tests
- `test_cpu_rope.py`: Cross-device compatibility tests

All tests verify:
- Correct tensor shapes
- Mathematical properties (magnitude preservation, invertibility)
- Positional differentiation
- Training compatibility
- Device handling

## References

- Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- The implementation follows the mathematical formulation from the original paper
