# CBOW based wordtovec
# Training sentence
text = "I love Python programming"

# Tokenize it:
token=text.lower().split()
print("Token:",token)

#Create vocabulary
vocabulary=sorted(set(token))
print("Vocabulary:",vocabulary)
wordtoid={word:id for id,word in enumerate(vocabulary)}
idtoword={id:word for word,id in wordtoid.items()}

# Convert the sentence to IDs:
token_ids=[wordtoid[i] for i in token]
print("token length",len(token_ids))
print("Token_IDS:",token_ids)

# Create CBOW training data 
# in cbow one token is target and words around the target is context
# for Love: context = [I, Python] and target = love
# for python: context = [love, programming] and target = Python
training_data=[]
windsize=1
for i in range(windsize,len(token_ids)-windsize): #for(1,3) 1,2
    context=[]
    for j in range(-windsize,windsize+1): #for(-1,2) 2 not consider here so -1,0,1
        if j !=0:
            context.append(token_ids[i+j])
    target=token_ids[i]
    training_data.append((context, target))
print(training_data)

# Now the actual neural network
import torch
import torch.nn as nn

# Create the CBOW model:
class CBOW(nn.Module):
    def __init__(self, vocabulary_size,embedding_dim):
        super().__init__()
        self.embedding= nn.Embedding(vocabulary_size,embedding_dim)

        self.linear = nn.Linear(embedding_dim,vocabulary_size)

    def forward(self,context):
        embeddings=self.embedding(context)
        avarage_embedding=embeddings.mean(dim=0)
        output=self.linear(avarage_embedding)
        return output

# Training
vocab_size=len(vocabulary)
embedding_dim=10

model=CBOW(vocab_size,embedding_dim)
loss_fuction=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(1000):
    total_loss=0
    for context,target in training_data:
        context_tensor=torch.tensor(context,dtype=torch.long)
        target_tensor=torch.tensor(target,dtype=torch.long)
        
        output=model(context_tensor)

        loss = loss_fuction(output.unsqueeze(0),target_tensor.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    if epoch % 100 == 0:
        print("Epoch:",epoch,"Loss:",total_loss)

embedding_matrix = model.embedding.weight # contains the learned word vectors.
print(embedding_matrix)

# Get the vector for Python
python_id = wordtoid["python"]
python_vector = model.embedding.weight[python_id]
print(python_vector)

# testing model
context = ["i","love","python"]
# words → IDs:
context_ids = [wordtoid[word] for word in context]
# tensor
context_tensor = torch.tensor(context_ids, dtype=torch.long)
output = model(context_tensor)
predicted_id = torch.argmax(output).item()
predicted_word = idtoword[predicted_id]
print(predicted_word)