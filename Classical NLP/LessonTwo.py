# i -> 17
# love -> 431
# machine -> 928

# are these number meaningful? No
# eg 431 is not mathematically saying that 'love' is not related to 'happy'

# ONE-Hot Encoding 

# we will go from token - tokenid - one hot vector - embedding - meaningful dense vector
# ONE-HOT Encoding
# only one postion matters in vector [0,0,0,0,0,1,0]
# if love -> 4 then vector is [0,0,0,0,1,0,0] for vcabulary size = 7
# python implementation
vocabsize=7
tokenid=4
onehot=[0]*vocabsize
onehot[tokenid]=1
print(onehot)

# for vocabsize = 5000 #imagine
# every token gets 5000 vector length
# almost everything is 0 this is called sparse representation
# and one hot vectors dont capture relationships
# eg:

# king
# queen
# man
# woman

# king   → [0.82, 0.31, -0.17, ...] 
# queen  → [0.79, 0.35, -0.15, ...] queen vector is more related to king vector becouse king and queen are realted
# man    → [0.51, 0.12, -0.30, ...]
# woman  → [0.49, 0.16, -0.28, ...]
# so we need embedding

# What is an embedding?
# An embedding is a dense numerical vector representing a token.
# Instead of: love → 4
# we eventually have something like: love → [0.21, -0.73, 0.44, 0.18, ...]
# Maybe 256 numbers Or 768 Or 4096.
# The number of dimensions is called the embedding dimension.

# One-hot
# [0, 1, 0, 0]
#      │
#      │ selects row 1
#      ▼

# Embedding Matrix

# [0.12, 0.45, 0.91]  ← token 0
# [0.92, 0.67, 0.45]  ← token 1  ✅
# [0.31, 0.88, 0.14]  ← token 2
# [0.72, 0.21, 0.63]  ← token 3

# Here dimension is 3
#                          ↓

#               [0.92, 0.67, 0.45]

# Eg from gpt based :
# EMBED_DIM = 256
# VOCAB_SIZE = 8000
# nn.Embedding(8000, 256)

# 8000 X 256 = 2,048,000 parameters.

#              256 dimensions
#          ┌──────────────────────┐
# Token 0  │ 0.12 -0.32 ... 0.71  │
# Token 1  │ 0.44  0.11 ... 0.23  │
# Token 2  │-0.21  0.82 ... 0.15  │
# Token 3  │ 0.31 -0.42 ... 0.91  │
#  ...     │          ...         │
# Token7999│0.08  0.51 ...-0.17   │
#          └──────────────────────┘

import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=7,embedding_dim=4)

#input
token_ids = torch.tensor([0,4,5,3])
vectors = embedding(token_ids)
print(vectors)
print(vectors.shape)

#        d1    d2    d3    d4
#      ┌──────────────────────
# 0    │ 0.2  0.1 -0.3  0.7
# 1    │-0.4  0.5  0.2  0.1
# 2    │ 0.7 -0.2  0.4  0.3
# 3    │ 0.1  0.8 -0.1  0.6
# 4    │-0.3  0.2  0.9 -0.4
# 5    │ 0.6 -0.5  0.1  0.2
# 6    │ 0.4  0.3 -0.7  0.5

# input [0, 4, 5, 3]

# 0 → row 0
# 4 → row 4
# 5 → row 5
# 3 → row 3

# [
#  [ 0.2,  0.1, -0.3,  0.7],
#  [-0.3,  0.2,  0.9, -0.4],
#  [ 0.6, -0.5,  0.1,  0.2],
#  [ 0.1,  0.8, -0.1,  0.6]
# ]