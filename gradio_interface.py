import gradio as gr
import torch
from gpt import BigramLanguageModel
from data_loader import load_data, get_batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = 'data/Rick_lines.txt'
train_data, val_data, encoder, decoder, vocab_size = load_data(data_path, min_freq=0)
model = BigramLanguageModel()
model.load_state_dict(torch.load('checkpoints/rick&morty_final.pth', map_location=torch.device('cpu')))
model.eval()
model = model.to(device)

def gpt_generate(prompt, n_words):
    enc_text = encoder(prompt)

    idx = torch.tensor(enc_text, dtype=torch.long)
    idx = idx.view(1, -1)
    idx = idx.to(device)

    output_text = decoder(model.generate(idx, max_new_tokens=n_words)[0].cpu().tolist())
    return output_text



demo = gr.Interface(fn=gpt_generate, inputs=["text", gr.Slider(0, 100)], outputs="text")

demo.launch()   
