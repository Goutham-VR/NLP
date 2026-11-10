# Lesson 3 — Bag of Words (BoW)
# Before Word2Vec, RNNs, Attention, and Transformers, you should understand how traditional NLP converted text into numbers.
# Look at which words occur in a document and count how many times each word occurs.
# It is called a bag because we mostly ignore word order.

#create token
documents = [
    "I love Python",
    "I love machine learning",
    "Python is powerful"
]

tokenized_doc=[]

for i in documents:
    tokens=i.split()
    tokenized_doc.append(tokens)
print(tokenized_doc)

#create vocab:
voc=set()
for i in tokenized_doc:
    for j in i:
        voc.add(j)
vocab=sorted(voc)
print(vocab)

# Now create the vectors:
bow_vectors=[]
for tokens in tokenized_doc:
    vector=[]
    for word in vocab:
        count=tokens.count(word)
        vector.append(count)
    bow_vectors.append(vector)
print(bow_vectors)


# Problem of BoW
# I love Python
# Python love I
# these two vectors are same = [1,1,1] 
# It doesn't understand: 'I love Python' vs 'Python love I'

# Why did we need BoW if we already had one-hot? They are related, but they're not the same.

# One-hot Represents one token:
# love -> [0,0,1,0,0]

# BoW Represents an entire document/sentence using word counts:
# I love Python -> [1,1,1,0,0]

# So:

# One-hot → token representation
# BoW → document representation

# N-grams
# BoW loses word order. So we introduce N-grams.
text = "I love Python"
tokens=text.split()
bigrams=[]
for i in range(len(tokens)-1):
    bigram=(tokens[i],tokens[i+1])
    bigrams.append(bigram)
print("Bigrams:",bigrams)

# For Trigram
tokens=text.split()
trigrams=[]
for i in range(len(tokens)-2):
    trigram=(tokens[i],tokens[i+1],tokens[i+2])
    trigrams.append(trigram)
print(trigrams)

# But N-grams have a problem too
# Suppose:
# Vocabulary = 50,000
# Bigram combinations can theoretically reach: 50,000 × 50,000 = 2.5 billion possible combinations.
# Trigrams become even larger.
# So N-grams improve order information but can explode in size.

# TF-IDF======================================================================================================
# TF = Term Frequency
# IDF = Inverse Document Frequency
# Together: TF-IDF = TF × IDF

# TF — Term Frequency
# How frequently does a word appear inside one document?
# Example:
# "I love Python Python"
# Python appears 2 times.
# So Python has a higher TF than I.

# IDF — Inverse Document Frequency
# How common is this word across the whole collection of documents?
# Suppose:
# Document 1 → I love Python
# Document 2 → I love Django
# Document 3 → I love Java

# I appears in all 3 documents.
# So:
# I → very common → low IDF
# But:
# Python → only appears in Document 1
# So:
# Python → rare → higher IDF
# Therefore TF-IDF gives more importance to words that are:
# frequent in this document but uncommon across the whole corpus.

# TF -> "How important is this word inside THIS document?"
# IDF -> "How rare is this word across ALL documents?"
# TF-IDF -> "How useful is this word for identifying THIS document?"

# TF-IDF is NOT a step in your GPT pipeline.
# You are learning it because you're learning the evolution of NLP:
# Text
#  ↓
# Tokenizer
#  ↓
# Token IDs
#  ↓
# Embedding
#  ↓
# Transformer
#  ↓
# Next token
# TF===========================
# D1 = "I love Python"
# D2 = "I love Django"
# D3 = "Python is powerful"

# D1 = "I love Python"
# TF = Word Count/total words

# TF(Python, D1) = 1 / 3
#                 = 0.333

# TF(I, D1) = 1 / 3
#           = 0.333

# I       → 0.333
# love    → 0.333
# Python  → 0.333
# =============================

# IDF==========================
# D1 → yes
# D2 → no
# D3 → yes
# N = 3 total docs
# So Python appears in 2 documents.
# IDF = log(N / DF)
# IDF(Python)
# = log(3 / 2)
# ≈ 0.405
# ================================

# TF-IDF==========================
# TF-IDF = TF × IDF
# TF     = 0.333
# IDF    = 0.405

# TF-IDF = 0.333 × 0.405
#        ≈ 0.135
# Python in D1 → 0.135
# ================================

# Word2Vec
# BoW/TF-IDF still don't really understand that: 'king','queen','man','woman','have' relationships
# Word2Vec introduces learned word representations, which leads us much closer to the embeddings used in neural NLP and eventually LLMs.
# EG
# "The cat eats food"
# "The dog eats food"
# BoW/TF-IDF mainly sees word counts. It doesn't naturally learn cat ↔ dog as similar concepts.
# Word2Vec tries to learn this kind of relationship from context.

# The core idea is:
# Words
#  ↓
# Look at surrounding words
#  ↓
# Learn useful numerical vectors
#  ↓
# Similar-context words get similar vectors

# Word2Vec
#    │
#    ├── CBOW
#    │
#    └── Skip-gram
# 1. CBOW
# Context → Target word
# Example: "The cat drinks milk"
# Given: The    drinks
# predict: cat

# So:
# Context words -> Neural network -> Target word

# 2. Skip-gram
# The opposite Approch
# Given: cat
# predict: The drinks

# so:
# Target word -> Neural network -> Context words

# Word2Vec is one of the historical ways of learning such word vectors.That's why Word2Vec is important for your LLM learning.
# But there is an important difference: Word2Vec learns one relatively fixed vector per word.
#                                       Modern Transformers produce representations that depend on context.

# For example: "I went to the bank to deposit money" and "I sat beside the river bank"

# The word bank has the same Word2Vec-style word vector, but a Transformer can create a context-dependent representation.
# That's one of the major steps from Word2Vec → contextual embeddings → Transformers.

# |                   | CBOW                  | Skip-gram                 |
# | ----------------- | --------------------- | ------------------------- |
# | Input             | Context               | Target                    |
# | Output            | Target                | Context                   |
# | Direction         | Context → Target      | Target → Context          |
# | Example           | `cat + milk → drinks` | `drinks → cat, milk`      |
# | Training examples | Fewer                 | More                      |
# | Basic idea        | Predict missing word  | Predict surrounding words |

# how CBOW/Skip-gram actually use an embedding matrix and learn those vectors. That will connect directly to the nn.Embedding you already learned.

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
context = ["i", "python"]
# words → IDs:
context_ids = [wordtoid[word] for word in context]
# tensor
context_tensor = torch.tensor(context_ids, dtype=torch.long)
output = model(context_tensor)
predicted_id = torch.argmax(output).item()
predicted_word = idtoword[predicted_id]
print(predicted_word)
#==================================================================

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