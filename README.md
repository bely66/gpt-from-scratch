# gpt-from-scratch

## Dataset
We're using the tiny shakespeare dataset, which consists of all the writings by shakespeare.

The dataset is loaded, encoded and splitted into train and test using the `dataloader.py`

**Tokenization:** We use a character level tokenization, where the model learns to predict the next character based on the context of the previous characters.

**Batching Data:** We batch data in the shape (B, T, C), B is the batch size or number of text blocks, T is the text block length we choose, C is the number of Channels or encoding size and in our case that's 65 which is the number of unique characters and that varies for different tokenizers

**Inputs** --> If input shape is (B, T, C) = (3, 7, 65) <br>
```
[   [2, 5, 7, 1, 2, 9, 12],
    [3, 6, 7, 1, 2, 9, 12],
    [4, 7, 7, 1, 2, 9, 12]  ]
```

***Let's take the training process for this input:***
Starting from the 1st time step trying to predict the next and increases the context eventually:
Input -> `[[2], [3], [4]]`
Targets -> `[[5], [6], [7]]`

Input -> `[[2, 5], [3, 6], [4, 7]]`
Targets -> `[[7], [7], [7]]`

Input -> `[[2, 5, 7], [3, 6, 7], [4, 7, 7]]`
Targets -> `[[1], [1], [1]]`

Etc....

The block size in this case is the maximum time steps the inputs can have.

For example if we have a block size of 8, then the input T channel will be 8 and so on

This is a tunable parameter for our model as it has many advantages and disadvantages:

1. The model can see variable size of context if given 1 character up to block size characters it can predict the next character
2. The model won't be able to see more than the block size so if it's incoherent or too small then the model ability to predict will be hindered by this


## Modelling

### Bigram Model
a baseline model is used to demonstrate the evolution of character modeling from a simple N-gram model to a more complex model like GPT

### GPT
**Generative-Pretrained-Transformer**



### Notes

1. **Attention** is a communication mechanism where you have a number of nodes in a directed graph where you have
edges between them

Every node has a **KEY** which is what makes me interesting, a **QUERY** what makes me interested and a **VALUE** what i'll send you if we're a match.
Edges in the graph is weighted based on the similarity between **QUERY** and **KEY**.

2. **Self-Attention** has no sense of space, it acts on a set of vectors and you have to introduce their positions using positional encodings.

3. There's no communication between the batch dimensions, every example sent to the attention has no relationship to the tokens in other examples in the batch.

