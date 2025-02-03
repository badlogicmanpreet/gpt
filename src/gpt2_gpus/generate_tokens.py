import torch
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass
import os
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2") # get the GPT2 encoding
ddp_rank = 0

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_UNIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_UNIT = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x   

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear): # initialize linear layers
            std = 0.02
            if hasattr(module, 'GPT_SCALE_UNIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # scale by the number of layers (scale down the std)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding): # initialize embedding layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        B, T = idx.size() # batch size, sequence length
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings for the sequence length T (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings for the sequence length T (B, T, n_embd)        
        x = tok_emb + pos_emb # combine the token and position embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # cross entropy loss
        return logits, loss


model = GPT(GPTConfig(vocab_size=50304))
checkpoint = torch.load('checkpoints/model_57218.pt', map_location=torch.device('cpu'))
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.eval() 

num_return_sequences = 4
max_length = 32
tokens = enc.encode("Training neural network is like,")
tokens = torch.tensor(tokens, dtype=torch.long) # convert to tensor
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # repeat for number of sequences (B, T)  
xgen = tokens.to(device) # move to GPU
sample_rng = torch.Generator(device=device) # create a generator, random number generator (rng), we dont want to disturb the global rng
sample_rng.manual_seed(42 + ddp_rank) # seed the rng
while xgen.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits, loss = model(xgen) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # logits is (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select the token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # ix is (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # xcol is (B, 1)
        # append the new token to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)
# print the generated sequences
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"rank {ddp_rank} sample {i}: {decoded}")

