
import torch
from gpt import *
from params import Params
from tokenizer import *

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('german_presidents.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer
chars = sorted(list(set(text)))
Params.vocab_size = len(chars)
tokenizer = TokenizerSimple(chars)

#Params.vocab_size = 1024
#tokenizer = Tokenizer()

# Params
device = Params.device

m = GPTModel()
m.load_state_dict(torch.load("gpt.model", map_location=device))
m.to(device)

# generate from the model
print("Generating ...")
context = torch.tensor(tokenizer.encode("Heute ist ein "), device=device).unsqueeze(0) 
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))

context = torch.zeros((1, 10), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
