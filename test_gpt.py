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

input_text = 'lives my'
enc_text = encoder(input_text)
print(enc_text)
idx = torch.tensor(enc_text, dtype=torch.long)
idx = idx.view(1, -1)
idx = idx.to(device)
print("Generation Result with training:")
output_text = decoder(model.generate(idx, max_new_tokens=10)[0].cpu().tolist())
print(output_text)