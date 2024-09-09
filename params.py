import torch
import tokenmonster


class Params:

    # hyperparameters
    batch_size = 48 # how many independent sequences will we process in parallel?
    block_size = 512 # what is the maximum context length for predictions?
    max_iters = 8000
    eval_interval = 500
    learning_rate = 2e-4
    device = torch.device('cpu')
    eval_iters = 200
    n_embd =  1024
    n_head = 8
    n_layer = 8
    dropout = 0.1
    vocab_size = 1024

    @classmethod
    def initialize(cls):
        cls.device = cls.autoselectDevice(verbose=1)

    # Check the devices that we have available and prefer CUDA over MPS and CPU
    @classmethod
    def autoselectDevice(cls, verbose=1):

        # default: CPU
        device = torch.device('cpu')

        if torch.cuda.is_available():
            # CUDA
            device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # MPS (acceleration on Apple silicon M1 / M2 chips)
            device = torch.device('mps')

        if verbose:
            print('Using device:', device)

        # Additional Info when using cuda
        if verbose and device.type == 'cuda':
            print(torch.cuda.get_device_name(0))

        return device

# Initialize the class attributes
Params.initialize()
