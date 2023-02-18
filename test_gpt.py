import torch
from gpt import BigramLanguageModel
from data_loader import load_data, get_batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = 'input.txt'
train_data, val_data, encoder, decoder, vocab_size = load_data(data_path)
model = BigramLanguageModel()
model.load_state_dict(torch.load('checkpoints/final.pth', map_location=torch.device('cpu')))
model.eval()
model = model.to(device)
print(model)

idx = torch.zeros((1, 1), dtype=torch.long)
idx = idx.to(device)
print("Generation Result with training:")
print(decoder(model.generate(idx, max_new_tokens=300)[0].cpu().tolist()))