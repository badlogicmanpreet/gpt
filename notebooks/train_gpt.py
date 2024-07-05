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
import math

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
        
        # qkv is the query, key, value projections
        # qkv is a tensor of shape (B, T, 3 * C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # q, k, v are tensors of shape (B, T, C)
        # TODO: (B, nh, T, hs) is the shape of q, k, v after the view and transpose operations below
        # B, nh -> batch size, number of heads, this is treated as batch size together.
        # Operations performed for all batches and all heads in parallel.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Following operation will be applied in parallel to all batches and all heads above
        # attention calculation (QK^T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # mask out the lower half of the dot product matrix, ensuring that q can only attend to k up to the current position
        # no looking into the future
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # softmax - normalizing the attention weights, sums up to 1 each row
        att = F.softmax(att, dim=-1)
        # attention * values -> basically a weighted sum of the values, using the attention weights
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        # reassamble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # final projection
        y = self.c_proj(y)
        return y

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
    # ***Remember that in normal transforer architecture, the layer normalization is
    # after the self attention or MLP and has layer normalization in its residual path,
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
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE tokens + 256 byte token + 1 special token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

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


    def forward(self, idx):
        B, T = idx.size() # batch size, sequence length
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings for the sequence length T (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings for the sequence length T (B, T, n_embd)
        x = tok_emb + pos_emb # combine the token and position embeddings
        # forward the block of transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer normalization and the linear layer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits


    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained model weights from Huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading {model_type} weights...")

        # instantiate a GPT2 model from Huggingface
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M Param
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M Param
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M Param
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M Param
            } [model_type]
        config_args['vocab_size'] = 50257 # GPT2 vocab size
        config_args['block_size'] = 1024 # GPT2 block size

        # create a from scratch minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # remove attention bias keys

        # initialize a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned correctly and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # remove attention bias keys
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # remove attention bias keys
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

# --------------------------------------------  -------------------------------------------- #
num_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('gpt2') # 124M 
model = GPT(GPTConfig()) # 124M
model.eval() # set to eval mode, model is not in training, we will just be using it for inference
#model.to('cuda') # move the model to the GPU on Cloudbox

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # convert to tensor (8,)
x = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # repeat for number of sequences (5, 8) (B, T)
#x = tokens.to('cuda') # move to GPU

# generate! right now x is (B, T) where B is the number of sequences to generate (5, 8)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # logits is (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # logits is (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select the token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # ix is (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # xcol is (B, 1)
        # append the new token to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated sequences
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

        

