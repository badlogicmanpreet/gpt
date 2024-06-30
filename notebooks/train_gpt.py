# GPT2 is a decoder only model, encoder and the cross attention piece is completely removed - https://arxiv.org/pdf/1706.03762
# In GPT2, https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
  #Layer normalization (Ba et al., 2016)
  #was moved to the input of each sub-block, similar to a
  #pre-activation residual network (He et al., 2016) and an
  #additional layer normalization was added after the final selfattention block. 

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    # Go back into my previous notes and understand the self attention mechanism
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is the number of heads, hs is the head size, and C (number of channels) = nh * hs
        # e.g in GPT2 (124M), n_head=12, hs=64, so nh*hs = 768 channels in the transformer

        




class MLP(nn.Module):
    # MLP is a simple feed forward network with GELU activation
    # GELU is a smooth version of ReLU, its tail end is not straight line like ReLU
    # but a curve, this helps in better gradient flow and learning. https://arxiv.org/pdf/1606.08415
    # GELU is also used in BERT and GPT2
    # RELU had a problem of dying neurons, where the neurons will not learn anything, 
    #      this is because the gradient of the ReLU is 0 for all negative values (flat region),
    #      and the neuron will not learn anything. GELU solves this problem by having a smooth
    #      curve at the tail end, which helps in better gradient flow.
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        

class Block(nn.Module):
    # Block is always the combination of self attention and MLP
    # along with layer normalization and residual connections
    # ***Remember that in normal transforer architecture, the layer normalization
    # after the self attention or MLP and has layer normalization in its path,
    # but in GPT2 the layer normalization is moved to the input of each sub-block i.e.
    # before the self attention or MLP and residual path is clear.
    # It is always desirable to have clean residual path, from supervision down to the input/token
    # layer, flow of gradients is smooth. Beauty is that gradients from the top will directly flow
    # to the input layer, because add operation is distributing the gradients equally to all the paths.
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    # fyi: x + is the residual connection
    # fyi: mlp is the feed forward network
    # fyi: attn is the aggregation or pooling function, this is where all tokens communicate
    #      with each other and update their representations. Also knows as wiegthed sum or
    #      reduce operation.
    #      mlp on the other side is the token wise operation, where each token is updated. Tokens
    #      are not communicating with each other, they are updated independently. This is more like
    #      a map operation.
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # transformer container just like the original GPT model
        # Embedding is simple lookup table
        # h is a stack of transformer blocks
        # ln_f is the final layer normalization before the output layer
        # lm_head is the output layer
        # moduledict can be indexed like a dictionary e.g. self.transformer['wte']
        # modulelist can be indexed like a list e.g. self.transformer[0]
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)