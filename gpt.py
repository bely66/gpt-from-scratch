import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from data_loader import load_data, get_batch

torch.manual_seed(1337)

## Hyper-Parameters
vocab_size = 65
batch_size = 256
head_size = 16
block_size = 8
max_iters = 6000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

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

class Head(nn.Module):

    def __init__(self, head_size) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.drop_out = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, idx):
        B, T, C = idx.shape
        k = self.key(idx) # (B, T, h)
        q = self.value(idx) # (B, T, h)

        wei = q @ k.transpose(-2, -1) * (head_size**-0.5) # (B, T, h) * (B, h, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop_out(wei)
        v = self.value(idx)
        out = wei @ v # (B, T, T) * (B, T, h) --> (B, T, h)
        return out

class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x

class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadedAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # (B, T, vocab_size) --> (B, T, C)
        self.positional_embeddings_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.heads = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        # self.sa_heads = MultiHeadedAttention(4, n_embed // 4)
        # self.sa_heads = Head(n_embed)
        self.ff = FeedForward()

    def forward(self, idx, targets=None):
        B, T = idx.shape

        out = self.token_embedding_table(idx)
        pos = self.positional_embeddings_table(torch.arange(T, device=device))
        x = out + pos
        x = self.heads(x)
        x = self.ff(x)
        logits = self.lm_head(x)

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

    model = BigramLanguageModel()
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    n = BigramLanguageModel()

    

    optim = torch.optim.AdamW(m.parameters(), lr=1e-3)

    print("Training the model in progress:")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in tqdm(range(max_iters)):

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
    print("Generation Result with training:")
print(decoder(m.generate(idx, max_new_tokens=300)[0].cpu().tolist()))

idx = torch.zeros((1, 1), dtype=torch.long)
idx.to(device)
print("Generation Result without training:")
print(decoder(m.generate(idx, 100)[0].cpu().tolist()))
print('-------------------------------------------------------- \n')
