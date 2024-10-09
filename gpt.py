
import torch
import torch.nn as nn
from torch.nn import functional as F
from params import Params

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)   
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)    

    def forward(self, x):
        # residual connection
        x = x + self.sa(self.ln1(x)) # communication
        x = x + self.ffwd(self.ln2(x)) # computation
        return x


# self attention head
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        # params
        n_embd = Params.n_embd
        block_size = Params.block_size
        dropout = Params.dropout

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # this will not be trained
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()

        # params
        n_embd = Params.n_embd        
        dropout = Params.dropout

        # multiple different "communication channels", e.g. "I am vowel", "I am consonant
        # information exchanged in these channels is 8-dimensional
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)      
        out = self.dropout(self.proj(out))
        return out
    

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()

        # params
        n_embd = Params.n_embd        
        dropout = Params.dropout

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   
            nn.Dropout(dropout), 
        )

    def forward(self, x):
        return self.net(x)

# super simple bigram model
class GPTModel(nn.Module):

    def __init__(self):
        super().__init__()

        # params
        vocab_size = Params.vocab_size
        n_embd = Params.n_embd        
        block_size = Params.block_size
        n_head = Params.n_head
        n_layer = Params.n_layer

        # each token directly reads off the logits for the next token from a lookup table
        # we have another indirection in terms of an actual embedding via a linear layer
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # also embed position
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
       
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm        
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):

        # This essentially follows the Attention is All You Need paper

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb  = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb  = self.position_embedding_table(torch.arange(T, device=Params.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)   

        # "communication" and "computation"
        x = self.blocks(x) # apply several blocks of multi-head attention (B,T,C) 
        # final layer norm 
        x = self.ln_f(x) # (B,T,C)    
        # and logits
        logits   = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, tokenizer=None):
        # idx is (B, T) array of indices in the current context
        block_size = Params.block_size
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) 
            
            # print
            if tokenizer:
                print(tokenizer.decode(idx_next[0].tolist()), end="", flush=True)
                       
        return idx