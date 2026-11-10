#  Multi-Head Attention lets the model learn multiple attention patterns in parallel.
# Imagine the sentence:

# "I love Python because it is powerful."

# When processing this sentence, the model needs to learn different relationships between tokens.

# For example:

# love relates to I.
# Python relates to love.
# powerful describes something about Python.
# it may refer to Python.

# A single attention head learns one set of attention patterns. 
# It might learn to focus on one kind of relationship, 
# while other heads can learn different patterns.

#in gpt
# EMBED_DIM = 256
# NUM_HEADS = 4

# dhead=dmodel/h
# =256/4

# Each token initially has a 256-dimensional embedding.
# We split the projected features across four attention heads, each working with 64-dimensional Q, K, and V vectors.
# | Component                     | Shape           |
# | ----------------------------- | --------------- |
# | Input embeddings              | `[B, T, 256]`   |
# | Q projection before splitting | `[B, T, 256]`   |
# | K projection before splitting | `[B, T, 256]`   |
# | V projection before splitting | `[B, T, 256]`   |
# | Q after splitting into heads  | `[B, 4, T, 64]` |
# | K after splitting into heads  | `[B, 4, T, 64]` |
# | V after splitting into heads  | `[B, 4, T, 64]` |
# | Attention scores              | `[B, 4, T, T]`  |
# | Each head's output            | `[B, 4, T, 64]` |
# | Concatenated output           | `[B, T, 256]`   |
# | Final output projection       | `[B, T, 256]`   |

# Here:

# B = batch size
# T = sequence length
# 256 = embedding dimension
# 4 = number of heads
# 64 = dimension per head

# Important: The model does not simply divide one attention result into four pieces. Each head has independent learnable projections and calculates its own attention.
# This example includes:

# Text tokenization
# Token IDs
# Token embeddings
# Q, K, and V projections
# Splitting into multiple heads
# Scaled dot-product attention
# Causal masking
# Concatenating the heads
# Output projection

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

tokens = text.lower().split()

# ============================================================
# 3. VOCABULARY AND TOKEN IDS
# ============================================================

vocab = sorted(set(tokens))

word_to_id = {
    word: idx
    for idx, word in enumerate(vocab)
}

token_ids = [word_to_id[token] for token in tokens]

input_ids = torch.tensor(token_ids, dtype=torch.long)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Input shape:", input_ids.shape)


# ============================================================
# 4. CONFIGURATION
# ============================================================

vocab_size = len(vocab)

embed_dim = 8
num_heads = 2

head_dim = embed_dim // num_heads

assert embed_dim % num_heads == 0

print("\nEmbedding dimension:", embed_dim)
print("Number of heads:", num_heads)
print("Dimension per head:", head_dim)


# ============================================================
# 5. TOKEN EMBEDDINGS
# ============================================================

embedding_layer = nn.Embedding(vocab_size, embed_dim)

# Shape: [T, E]
x = embedding_layer(input_ids)

# Add batch dimension.
# Shape: [B, T, E]
x = x.unsqueeze(0)

print("\nEmbedding shape:", x.shape)


# ============================================================
# 6. MULTI-HEAD ATTENTION CLASS
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projection layers.
        # Each projection maps E features to E features.

        self.query_layer = nn.Linear(
            embed_dim, embed_dim, bias=False
        )

        self.key_layer = nn.Linear(
            embed_dim, embed_dim, bias=False
        )

        self.value_layer = nn.Linear(
            embed_dim, embed_dim, bias=False
        )

        # Combines information from all heads.

        self.output_layer = nn.Linear(
            embed_dim, embed_dim, bias=False
        )


    def forward(self, x):
        # x shape: [B, T, E]
        batch_size, seq_len, embed_dim = x.shape
        # ----------------------------------------------------
        # STEP 1: CREATE Q, K, V
        # ----------------------------------------------------

        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        # Shapes: [B, T, E]

        print("\nQ before splitting:", Q.shape)
        print("K before splitting:", K.shape)
        print("V before splitting:", V.shape)

        # ----------------------------------------------------
        # STEP 2: SPLIT EMBEDDING FEATURES INTO HEADS
        # ----------------------------------------------------
        # ഓരോ token-ന്റെയും 8 features-നെ രണ്ട് heads-ലായി 4 വീതം വിഭജിക്കണം.
        Q = Q.reshape(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        )

        K = K.reshape(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        )

        V = V.reshape(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        )

        # Shapes: [B, T, H, D]

        # Move heads before sequence dimension:
        # [B, T, H, D] -> [B, H, T, D]
        # Attention scores കണക്കാക്കാൻ ഓരോ head-നും അതിന്റെ എല്ലാ tokens-ന്റെയും Q, K vectors വേണം. thats why we use transpose(1, 2)

        Q = Q.transpose(1, 2) # [B, H, T, D]
        K = K.transpose(1, 2) # [B, H, T, D]
        V = V.transpose(1, 2) # [B, H, T, D]

        print("\nQ after splitting:", Q.shape)
        print("K after splitting:", K.shape)
        print("V after splitting:", V.shape)

        # ----------------------------------------------------
        # STEP 3: CALCULATE ATTENTION SCORES
        # ----------------------------------------------------

        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Shape: [B, H, T, T] 
        # becouse [.,.,T,D] of Q X [.,.,D,T] of KTranspose = T X T ; Inner dimensions D cancel out
        # 1st T = Query Token
        # 2nd T = Key Token

        scores = scores / math.sqrt(self.head_dim)
        print("\nAttention scores shape:", scores.shape)

        # ----------------------------------------------------
        # STEP 4: APPLY CAUSAL MASK
        # ----------------------------------------------------

        causal_mask = torch.tril(
            torch.ones(
                seq_len, seq_len,
                dtype=torch.bool,
                device=x.device # for use of CPU
            )
        )

        # Broadcast the [T, T] mask over batches and heads.
        # Future positions receive -infinity.
        scores = scores.masked_fill(~causal_mask,float("-inf"))

        # ----------------------------------------------------
        # STEP 5: SOFTMAX
        # ----------------------------------------------------
        attention_weights = F.softmax(scores, dim=-1)
        print("\nAttention weights shape:",attention_weights.shape)

        # ----------------------------------------------------
        # STEP 6: ATTENTION WEIGHTS @ V
        # ----------------------------------------------------
        output = torch.matmul(attention_weights, V)
        # Shape: [B, H, T, D]
        print("Each head's output shape:", output.shape)

        # ----------------------------------------------------
        # STEP 7: COMBINE THE HEADS
        # ----------------------------------------------------
        # [B, H, T, D] -> [B, T, H, D]
        output = output.transpose(1, 2).contiguous()
        # [B, T, H, D] -> [B, T, E]
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        print("Concatenated output shape:", output.shape)

        # ----------------------------------------------------
        # STEP 8: OUTPUT PROJECTION
        # ----------------------------------------------------
        output = self.output_layer(output)
        print("Final output shape:", output.shape)
        return output


# ============================================================
# 7. RUN MULTI-HEAD ATTENTION
# ============================================================

model = MultiHeadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads
)

output = model(x)

print("\nInput shape:", x.shape)
print("Final MHA output shape:", output.shape)
print("\nFinal output:")
print(output)