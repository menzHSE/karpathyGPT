import argparse
import torch
from gpt import *
from params import Params
from tokenizer import *

# Argument parser setup
parser = argparse.ArgumentParser(description="Load a GPT model")
parser.add_argument("-m", "--model", type=str, required=False, default="gpt.model",
                    help="Path to the model file (default: gpt.model)")
args = parser.parse_args()



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

# Load the model
m = GPTModel()
model_path = args.model  # model path from command line argument
m.load_state_dict(torch.load(model_path, map_location=device))
m.to(device)


# generate from the model
print("Generating ...")
context = torch.tensor(tokenizer.encode("Heute ist ein Tag der "), device=device).unsqueeze(0) 
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))

context = torch.zeros((1, 10), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
