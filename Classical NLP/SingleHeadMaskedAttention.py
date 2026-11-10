
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. TEXT
# ============================================================

text = "I love Python"

# ============================================================
# 2. TOKENIZATION
# ============================================================
# Simple tokenizer for learning.
# Your ModernGPT uses ByteLevel BPE instead.

tokens = text.lower().split()

# ============================================================
# 3. VOCABULARY AND TOKEN IDs
# ============================================================

vocab = sorted(set(tokens))

word_to_id = {
    word: idx
    for idx, word in enumerate(vocab)
}

id_to_word = {
    idx: word
    for word, idx in word_to_id.items()
}

token_ids = [word_to_id[word] for word in tokens]

print("Text:", text)
print("Tokens:", tokens)
print("Vocabulary:", vocab)
print("Token IDs:", token_ids)

# ============================================================
# 4. TOKEN IDs -> TENSOR
# ============================================================

input_ids = torch.tensor(token_ids, dtype=torch.long)

print("\nInput IDs:")
print(input_ids)
print("Input shape:", input_ids.shape)  # [T] = [3]

# ============================================================
# 5. TOKEN EMBEDDINGS
# ============================================================

vocab_size = len(vocab)
embed_dim = 4

embedding_layer = nn.Embedding(vocab_size, embed_dim)
x = embedding_layer(input_ids)

print("\nToken embeddings:")
print(x)
print("Embedding shape:", x.shape)  # [T, E] = [3, 4]

# ============================================================
# 6. QUERY, KEY, VALUE PROJECTIONS
# ============================================================

query_layer = nn.Linear(embed_dim, embed_dim, bias=False)
key_layer = nn.Linear(embed_dim, embed_dim, bias=False)
value_layer = nn.Linear(embed_dim, embed_dim, bias=False)

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

print("\nQ shape:", Q.shape)  # [3, 4]
print("K shape:", K.shape)  # [3, 4]
print("V shape:", V.shape)  # [3, 4]

# ============================================================
# 7. ATTENTION SCORES: Q @ K^T
# ============================================================

scores = torch.matmul(Q, K.transpose(-2, -1))

print("\nRaw attention scores:")
print(scores)
print("Scores shape:", scores.shape)  # [T, T] = [3, 3]

# ============================================================
# 8. SCALE THE SCORES
# ============================================================
# d_k = dimension of key
d_k = K.size(-1)
scores = scores / math.sqrt(d_k)

print("\nScaled attention scores:")
print(scores)

# ============================================================
# 9. CAUSAL MASK
# ============================================================
# True  = token is allowed to attend
# False = future token is blocked

seq_len = input_ids.size(0)
causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

print("\nCausal mask:")
print(causal_mask.int())

# ============================================================
# 10. BLOCK FUTURE TOKENS
# ============================================================

masked_scores = scores.masked_fill(~causal_mask,float("-inf"))

print("\nMasked attention scores:")
print(masked_scores)

# ============================================================
# 11. SOFTMAX -> ATTENTION WEIGHTS
# ============================================================

attention_weights = F.softmax(masked_scores, dim=-1)

print("\nAttention weights:")
print(attention_weights)

print("Sum of each row:",attention_weights.sum(dim=-1))

# ============================================================
# 12. ATTENTION WEIGHTS @ VALUES
# ============================================================

output = torch.matmul(attention_weights, V)

print("\nFinal attention output:")
print(output)

print("Output shape:", output.shape)  # [T, E] = [3, 4]

# ============================================================
# 13. VERIFY FUTURE TOKENS ARE BLOCKED
# ============================================================

future_weights = attention_weights.masked_select(~causal_mask)

print("\nAttention weights assigned to future tokens:")
print(future_weights)

# They should all be zero.
print("Are all future weights zero?",torch.all(future_weights == 0).item())