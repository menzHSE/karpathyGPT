import argparse
import torch
from gpt import *
from params import Params
from tokenizer import *

# Argument parser setup
parser = argparse.ArgumentParser(description="Load a GPT model")
parser.add_argument("-m", "--model", type=str, required=False, default="gpt.model",
                    help="Path to the model file (default: gpt.model)")
parser.add_argument("-c", "--context", type=str, default=".",
                    help="Initial context text (default: '.')")
parser.add_argument("-n", "--numTokens", type=int, default=100,
                    help="Number of tokens to generate (default: 100)")
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
contextText = args.context
numTokens = args.numTokens
print(f'Generating {numTokens} with context "{contextText}" ...')


# with context
context = torch.tensor(tokenizer.encode(contextText), device=device).unsqueeze(0) 
print(tokenizer.decode(m.generate(context, max_new_tokens=numTokens)[0].tolist()))

