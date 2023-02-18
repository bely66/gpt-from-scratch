import torch
import math


device = torch.device("cpu")
from tqdm import tqdm
from bigram import BigramLanguageModel
from data_loader import load_data, get_batch
torch.manual_seed(1337)


## Hyper-Parameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, block_size, batch_size,split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

data_path = 'input.txt'
train_data, val_data, encoder, decoder, vocab_size = load_data(data_path)
block_size = 8

model = BigramLanguageModel(vocab_size)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

n = BigramLanguageModel(vocab_size)



optim = torch.optim.AdamW(m.parameters(), lr=1e-3)

print("Training the model in progress:")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch(train_data, val_data, block_size, batch_size,'train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1, 1), dtype=torch.long)
idx.to(device)
print("Generation Result without training:")
print(decoder(n.generate(idx, 100)[0].cpu().tolist()))
print('-------------------------------------------------------- \n')

print("Generation Result with training:")
print(decoder(m.generate(idx, max_new_tokens=300)[0].cpu().tolist()))