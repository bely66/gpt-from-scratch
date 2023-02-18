import torch
from torch import nn

from tqdm import tqdm
from data_loader import load_data, get_batch
torch.manual_seed(1337)

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 6000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
n_head = 6
n_layer = 6
dropout = 0.2
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = nn.functional.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # (B, T, vocab_size) --> (B, T, C)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embed)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeds = self.token_embedding_table(idx) # (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeds + pos_embed
        x = self.sa_head(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



if __name__ == "__main__":
    data_path = 'input.txt'
    train_data, val_data, encoder, decoder, vocab_size = load_data(data_path)

    m = BigramLanguageModel()

    

    optim = torch.optim.AdamW(m.parameters(), lr=1e-3)

    print("Training the model in progress:")
    for steps in tqdm(range(max_iters)):
        xb, yb = get_batch(train_data, val_data, block_size, batch_size, 'train')
        xb.to(device)
        yb.to(device)
        logits, loss = m(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if steps % 100 == 0:
            print(f"epcoh {steps}, loss= {loss.item()}")
    print(f"epcoh {steps}, loss= {loss.item()}")


idx = torch.zeros((1, 1), dtype=torch.long)
idx.to(device)
print("Generation Result with training:")
print(decoder(m.generate(idx, max_new_tokens=300)[0].cpu().tolist()))
