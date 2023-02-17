import torch
import math


device = torch.device("cpu")
from tqdm import tqdm
from bigram import BigramLanguageModel
from data_loader import load_data, get_batch
torch.manual_seed(1337)


data_path = 'input.txt'
train_data, val_data, encoder, decoder, vocab_size = load_data(data_path)
block_size = 8
n = BigramLanguageModel(vocab_size)
n.to(device)
m = BigramLanguageModel(vocab_size)
m.to(device)
optim = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
epochs = int(20e3)
print("Training the model in progress:")
for steps in tqdm(range(epochs)):
    xb, yb = get_batch(train_data, val_data, block_size, batch_size, 'train')
    xb.to(device)
    yb.to(device)
    logits, loss = m(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

print(f"epcoh {steps}, loss= {loss.item()}")

idx = torch.zeros((1, 1), dtype=torch.long)
idx.to(device)
print("Generation Result without training:")
print(decoder(n.generate(idx, 100)[0].cpu().tolist()))
print('-------------------------------------------------------- \n')

print("Generation Result with training:")
print(decoder(m.generate(idx, max_new_tokens=300)[0].cpu().tolist()))