# Causal Masking in GPT
# When training GPT to predict the next token, it should learn:
# I → predict love
# I love → predict Python
# I love Python → predict programming
# The model must predict the next token without seeing future tokens.


import torch

# Three tokens in our input sequence
tokens = ["I", "love", "Python"]

seq_len = len(tokens)

# Create a lower-triangular matrix
mask = torch.tril(torch.ones(seq_len, seq_len))

print(mask)
# torch.tril() keeps the lower triangle and sets the upper triangle to zero.
# 1 means attention is allowed.
# 0 means attention is blocked.

# Apply the mask to attention scores
# In real attention, we calculate scores using: Q × Kᵀ/sqrt(dk)
# Then we block future-token scores before applying softmax.


import torch
import torch.nn.functional as F
import math

tokens = ["I", "love", "Python"]
seq_len = len(tokens)
d_k = 4

# Example attention scores (normally calculated from Q and K)
scores = torch.tensor([
    [2.0, 3.0, 4.0],
    [1.0, 2.0, 5.0],
    [3.0, 2.0, 1.0]
])

# Create causal mask
mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

# Block future-token attention scores
scores = scores.masked_fill(~mask, float("-inf"))

# Convert scores into attention weights
attention_weights = F.softmax(scores, dim=-1)

print("Attention weights:")
print(attention_weights)