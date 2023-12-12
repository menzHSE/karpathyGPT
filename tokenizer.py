import tokenmonster
from params import Params

class Tokenizer:
    def __init__(self):               
        tokenmonster.set_local_directory("./_tokenmonster")
        # Load a vocabulary by name, filepath or URL
        self.ttenc = tokenmonster.load("english-" + str(Params.vocab_size) + "-consistent-v1")

    def encode(self, s):
        return list(self.ttenc.tokenize(s))
    
    def decode(self, l):
        return self.ttenc.decode(l)
    

class TokenizerSimple:
    def __init__(self, chars):               
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string