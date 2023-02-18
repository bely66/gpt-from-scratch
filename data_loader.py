import torch
import spacy
from collections import Counter

def read_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def load_data(file_path, min_freq=3):
    text = read_file(file_path)
    word2id = {'<PAD>':0, '<UNK>':1}
    id2word = {0:'<PAD>', 1:'<UNK>'}
    word_freq = {}

    wc = Counter()
    for word in text.strip().split(' '):
        wc[word.lower()] += 1

    for word, count in wc.most_common():
        if count < min_freq: break
        wid = len(word2id)
        word2id[word] = wid
        id2word[wid] = word

    encode = lambda s: [word2id[c.lower()] if c in word2id else word2id['<UNK>'] for c in s.strip().split()]
    decode = lambda e: ' '.join([id2word[i] for i in e])
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data = split_data(data)
    return train_data, val_data, encode, decode, len(word2id)


def split_data(data):
    n = int(0.9*data.shape[0])
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(train_data, val_data, block_size, batch_size, split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y


if __name__ == "__main__":
    file_name = 'input.txt'
    block_size = 8 # what's the maximum context length for predictions?
    batch_size = 4 # how many independent sequences will be processed in parallel?
    train, val, encoder, decoder, vocab_size = load_data(file_name)
    x, y = get_batch(train, val, block_size, batch_size, 'train')

    print('Inputs:')
    print(x.shape)
    print(x)

    print('Targets:')
    print(y.shape)
    print(y)
    # text = read_file(file_name)
    # chars = sorted(list(set(text)))
    # stoi = {s:i for i, s in enumerate(chars)}
    # itos = {i:s for i, s in enumerate(chars)}
    # encode = lambda s: [stoi[c] for c in s]
    # decode = lambda e: ''.join([itos[i] for i in e])

    # print(text[:100])
    # print(f'All chars in the dataset:{chars} \n and their length is:{len(chars)}')

    # print('Encoded Text:', encode(text[5:12]))
    # print('Decoded Text:', decode(encode(text[5:12])))


    # encode the data to a torch tensor

    # data = torch.tensor(encode(text), dtype=torch.long)
    # print(data.shape)
    # print(data[:10])

    # n = int(0.9*data.shape[0])
    # train_data = data[:n]
    # val_data = data[n:]

    # # batches
    # block_size = 8
    # x = train_data[:block_size]
    # y = train_data[1:block_size+1]
    # for t in range(block_size):
    #     context = x[:t+1]
    #     target = y[t]
    #     print(f'when input is {context} the target is {target}')


