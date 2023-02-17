import torch
from torch import nn

from data_loader import load_data, get_batch
torch.manual_seed(1337)
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # (B, T, vocab_size) --> (B, T, C)

    def forward(self, idx, targets):

        logits = self.token_embedding_table(idx)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss
    

data_path = 'input.txt'
train_data, val_data, encoder, decoder, vocab_size = load_data(data_path)
batch_size = 4
block_size = 8
xb, yb = get_batch(train_data, val_data, block_size, batch_size, 'train')
m = BigramLanguageModel(vocab_size)

out, loss = m(xb, yb)
print(out.shape)
print(loss)