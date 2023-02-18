import torch
from torch import nn

from tqdm import tqdm
from data_loader import load_data, get_batch
torch.manual_seed(1337)

device = torch.device("cpu")

## Hyper Parameters
batch_size = 32
max_iters = int(20e3)
eval_interval = 300
learning_rate = 1e-2
block_size = 8
n_embed = 32

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # (B, T, vocab_size) --> (B, T, C)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeds = self.token_embedding_table(idx) # (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeds + pos_embed
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
            logits, loss = self(idx)
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
    print(f"epcoh {steps}, loss= {loss.item()}")


