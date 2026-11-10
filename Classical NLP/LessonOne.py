# What is a token?
# A token is a piece of text that the NLP system treats as one unit.

# For example: I love Python
# could become: ["I", "love", "Python"]

# Here we have 3 tokens.

# But tokens don't necessarily have to be complete words.
# For example: 'unbelievable' could be split into:

# ["un", "believ", "able"]

# Or even into smaller pieces depending on the tokenizer.

# This is extremely important for understanding GPT.

# 4. Word tokenization

# The simplest tokenizer is word-based tokenization.

# Example:

# I love machine learning

# becomes:

# ["I", "love", "machine", "learning"]

# eg code:
text = "I love Machine Learning"
token=text.split()
print(token)

# Output
['I', 'love', 'Machine', 'Learning']

# Why can't we just give these words to a neural network?
# But neural networks need numbers.

# We could create a vocabulary.
# Suppose our corpus is:

# I love Python
# I love machine learning
# Python is powerful

# Our unique words might be:
# I
# love
# Python
# machine
# learning
# is
# powerful

# Now assign each word an ID:
# I         → 0
# love      → 1
# Python    → 2
# machine   → 3
# learning  → 4
# is        → 5
# powerful  → 6

# This is called a vocabulary.

# Then 'I love machine learning' becomes:

# 0 1 3 4

# Now the computer has numbers.
text = "i love python i love machine learning python is powerful"
tokens = text.split()

#  Create Vocabulary
vocab=sorted(set(tokens))
print(vocab)

token_to_id={
    token:index for index,token in enumerate(vocab)
}
print(token_to_id)

id_to_token={
    index:token for token,index in token_to_id.items()
}
print(id_to_token)

# Encoding
def encoder(text):
    tokens=text.split()
    ids=[
        token_to_id[token] for token in tokens
    ]
    return ids

text='i love'
en=encoder(text)
print(en)

# Decoding
def decoder(ids):
    tokens=[
        id_to_token[id] for id in ids
    ]
    return " ".join(tokens)

ids=[0,3]
ids=decoder(ids)
print(ids)

# later use BPETokenizer. split() is simple and initial step

# Probelm with word level tokenization
# 1. suppose input is 'i love artificial intelligence' this is not in our current vocabulary we need an unknown token so it become 'i love <UNK> <UNK>' this is not ideal

# 2. if we create a vocabulary conatin all possible english words its complicated

# 3. supercalifragilisticexpialidocious what if the model has never seen it? then word tokenizer struggles
# Here comes new method subword tokenizer

# SUBWORD TOKENIZER
# Modern NLP system tokenizer
# Instead of treating every word as one indivisible unit, words can be broken into smaller pieces
# 'playing' become play+ing
# unhappiness become un+happiness or un+happi+ness depending on the tokenizer
# common algorithms are Byte Pair Encoding (BPE), WordPiece, Unigram, SentencePiece






