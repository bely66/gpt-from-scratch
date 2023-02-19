import torch
from time import time
torch.manual_seed(1337)
torch.set_printoptions(sci_mode=False)
B, T, C = 4, 8, 2

x = torch.randn(B, T, C)
print(x.shape)

# Mathematical Trick of Self Attention
## We want token in the T dimension to only talk to previous tokens (How can we do that mathematically?)
### Averaging of all the previous tokens would be the best way to do that

xbow = torch.randn(B, T, C)
start = time()
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # shape --> (t, C)
        xbow[b, t] = torch.mean(xprev, dim=0) # shape --> (C)
end = time()
print(f"For loop xbow took:{end - start}")
# The above way to calculate the average is inefficient
## How can we do that using Matrix Multiplication?
### We're calculating the average across the time dimension
#### So our averaging matrix should be doing the operation across the T dimension
##### Our averaging matrix will be T, T
start = time()

averaging_matrix = torch.tril(torch.ones(T, T))
averaging_matrix = averaging_matrix / torch.sum(averaging_matrix, dim=1, keepdim=True)

xbow2 = averaging_matrix @ x # (T, T) * (B, T, C) --> (B, T, T) * (B, T, C) --> (B, T, C)
end = time()
print(f"Matrix multiplication xbow took:{end - start}")
print(torch.allclose(xbow, xbow2))


## Version 3 Softmax
start = time()

trill_matrix = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)
wei = wei.masked_fill(trill_matrix == 0, float('-inf'))
wei = torch.nn.functional.softmax(wei, dim=-1)
xbow3 = wei @ x
end = time()

print(f"Matrix multiplication softmax xbow took:{end - start}")
print(torch.allclose(xbow3, xbow2))


# single head of self-attention
## this way the relationship between the token and the elements before becomes data depndant

head_size = 16
key = torch.nn.Linear(C, head_size, bias=False)
query = torch.nn.Linear(C, head_size, bias=False)
value = torch.nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)

wei = q @ k.transpose(-2, -1) # (B, T, 16) * (B, 16, T) -> (B, T, T)
wei = wei.masked_fill(trill_matrix == 0, float('-inf'))
wei = torch.nn.functional.softmax(wei, dim=-1)
v = value(x)
out = wei @ v

print(out.shape)


## Scaled Attention
print("----------------------------------------------")
print('')
print("Scaled Attention Section")
k = torch.randn((B, T, 16)) # (B, T, 16)
q = torch.randn((B, T, 16)) # (B, T, 16)
wei = q @ k.transpose(-2, -1) # (B, T, 16) * (B, 16, T) -> (B, T, T)
wei_s = torch.nn.functional.softmax(wei, dim=-1)
print(f"Before Normalization Variance of weights:{wei.var()}, keys:{k.var()}, query:{q.var()}")
print(f"Weights Before Normalization:{wei[0][0]}")
print(f"Softmax Before Normalization: {wei_s[0][0]} and sum: {wei_s[0][0].sum()}")
wei *= (16**-0.5)
wei_s = torch.nn.functional.softmax(wei, dim=-1)
print(f"After Normalization Variance of weights:{wei.var()}, keys:{k.var()}, query:{q.var()}")
print(f"Weights After Normalization:{wei[0][0]}")
print(f"Softmax After Normalization: {wei_s[0][0]} and sum: {wei_s[0][0].sum()}")
