#Skip-gram Wordtovec
text = "I love Python programming"

tokens = text.lower().split()

vocab = sorted(set(tokens))

word_to_id = {word: i for i, word in enumerate(vocab)}

id_to_word = {i: word for word, i in word_to_id.items()}

token_ids = [word_to_id[word]for word in tokens]

print(tokens)
print(word_to_id)
print(token_ids)

training_data=[]
windsize=1

for i in range(windsize,len(token_ids)-windsize):
    target=token_ids[i]
    for j in range(-windsize,windsize+1):
        if j!=0:
            context=token_ids[i+j]
            training_data.append((target,context))
print(training_data)

# Skip-Gram Model:
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self,vocabularysize,embeddingdim):
        super().__init__()
        self.embedding=nn.Embedding(vocabularysize,embeddingdim)
        self.linear=nn.Linear(embeddingdim,vocabularysize)
    def forward(self,target):
        embeddings=self.embedding(target)
        output=self.linear(embeddings)
        return output

vocabularysize=len(vocab)
embeddingdim=10
model=SkipGram(vocabularysize,embeddingdim)
lossfunction=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

for i in range(1000):
    totalloss=0
    for target,context in training_data:
        target_tensor=torch.tensor(target,dtype=torch.long)
        context_tensor=torch.tensor(context,dtype=torch.long)
        output=model(target_tensor)

        loss=lossfunction(output.unsqueeze(0),context_tensor.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalloss+=loss.item()

    if i % 100 == 0:
        print("Epoch:",i,"Loss:",totalloss)

# embeddingmatrix=model.embedding.weight()
# print("Embedding Matrix:",embeddingmatrix)

python_id = word_to_id["python"]

python_vector = model.embedding.weight[python_id]

print(python_vector)

# testing model
target = ["python"]
# words → IDs:
target_ids = [word_to_id[word] for word in target]
# tensor
target_tensor = torch.tensor(target_ids, dtype=torch.long)
output = model(target_tensor)
predicted_id = torch.argmax(output).item()
predicted_word = id_to_word[predicted_id]
print(predicted_word)