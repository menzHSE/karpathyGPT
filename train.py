# from gpt-dev.ipynb

import torch
import torch.nn as nn
from torch.nn import functional as F
from params import *
from gpt import *
from tokenizer import *

torch.manual_seed(1337)

# read it in to inspect it
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#with open('german_presidents.txt', 'r', encoding='utf-8') as f:
with open('speeches.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer
chars = sorted(list(set(text)))
Params.vocab_size = len(chars)
#tokenizer = TokenizerSimple(chars)

Params.vocab_size = 1024
tokenizer = Tokenizer()

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Params
block_size    = Params.block_size
batch_size    = Params.batch_size
device        = Params.device
eval_iters    = Params.eval_iters
learning_rate = Params.learning_rate
max_iters     = Params.max_iters
eval_interval = Params.eval_interval

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out




model = GPTModel()
#model.load_state_dict(torch.load("gpt.model", map_location=device))
model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save(model.state_dict(), "gpt.model." + str(iter))

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f".")
torch.save(model.state_dict(), "gpt.model")
