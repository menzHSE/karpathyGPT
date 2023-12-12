
import torch
from gpt import *
from params import Params
from tokenizer import *

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer
chars = sorted(list(set(text)))
Params.vocab_size = len(chars)
tokenizer = TokenizerSimple(chars)

# Params
device = Params.device

m = GPTModel()
m.load_state_dict(torch.load("gpt.model"))
m.to(device)

# generate from the model
print("Generating ...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
