# How can a neural network process a sequence of words while remembering previous words?
# ============================================================
# RNN - Next Token Prediction
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# 1. TEXT
# ============================================================

text = "I love Python programming"

tokens = text.lower().split()

print("Tokens:", tokens)


# ============================================================
# 2. VOCABULARY
# ============================================================

vocab = sorted(set(tokens))

word_to_id = {
    word: i
    for i, word in enumerate(vocab)
}

id_to_word = {
    i: word
    for word, i in word_to_id.items()
}

print("Word to ID:", word_to_id)
print("ID to Word:", id_to_word)


# ============================================================
# 3. WORDS → TOKEN IDs
# ============================================================

token_ids = [
    word_to_id[word]
    for word in tokens
]

print("Token IDs:", token_ids)


# ============================================================
# 4. CREATE RNN TRAINING DATA
# ============================================================

# RNN learns:
#
# I                  → love
# I love             → Python
# I love Python      → programming
#
# For simplicity, we will use one complete sequence:
#
# Input  = I love Python
# Target = love Python programming

input_ids = token_ids[:-1]
target_ids = token_ids[1:]

print("Input IDs:", input_ids)
print("Target IDs:", target_ids)

# ============================================================
# 5. CONVERT TO TENSORS
# ============================================================

input_tensor = torch.tensor([input_ids],dtype=torch.long)
target_tensor = torch.tensor([target_ids],dtype=torch.long)

print("Input Tensor:", input_tensor)
print("Target Tensor:", target_tensor)

print("Input Shape:", input_tensor.shape)
print("Target Shape:", target_tensor.shape)

# ============================================================
# 6. RNN MODEL
# ============================================================

class SimpleRNN(nn.Module):
    def __init__(self,vocabularysize,embeddingdim,hiddensize):
        super().__init__()
        # Token ID → Vector
        self.embedding = nn.Embedding(vocabularysize,embeddingdim)
        # Sequence processing
        self.rnn = nn.RNN(input_size=embeddingdim,hidden_size=hiddensize,batch_first=True)
        # Hidden state → Vocabulary scores
        self.linear = nn.Linear(hiddensize,vocabularysize)

    def forward(self, x):
        embeddings = self.embedding(x)
        output, hidden = self.rnn(embeddings)
        output = self.linear(output)
        return output


# ============================================================
# 7. CREATE MODEL
# ============================================================
vocabularysize = len(vocab)
embeddingdim = 10
hiddensize = 20
model = SimpleRNN(vocabularysize,embeddingdim,hiddensize)
lossfunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

# ============================================================
# 10. TRAINING
# ============================================================
for epoch in range(1000):
    # Forward pass
    output = model(input_tensor)
    # --------------------------------
    # Reshape output
    output = output.reshape(-1,vocabularysize)
    target = target_tensor.reshape(-1)
    # --------------------------------
    # Calculate loss
    loss = lossfunction(output,target)
    # --------------------------------
    # Clear old gradients
    optimizer.zero_grad()
    # --------------------------------
    # Backpropagation
    loss.backward()
    # --------------------------------
    # Update weights
    # --------------------------------
    optimizer.step()
    # --------------------------------
    # Print loss
    # --------------------------------
    if epoch % 100 == 0:
        print("Epoch:",epoch,"Loss:",loss.item())

# ============================================================
# 11. TEST / PREDICTION
# ============================================================
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    predicted_ids = torch.argmax(output,dim=-1)
print("Predicted IDs:", predicted_ids)


# ============================================================
# 12. IDs → WORDS
# ============================================================
# for rnn prediction for each postion in input sequence not next word
predicted_words = []

for predicted_id in predicted_ids[0]:

    word = id_to_word[
        predicted_id.item()
    ]

    predicted_words.append(word)


print("Predicted Words:", predicted_words)