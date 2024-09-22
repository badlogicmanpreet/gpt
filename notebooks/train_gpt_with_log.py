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
        assert config.n_embd % config.n_head == 0 # TODO: study this in detail
        # key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_UNIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    # Go back into my previous notes and understand the self attention mechanism
    def forward(self, x):
        logger.debug(f"causal self attention (forward section): x size: {x.size()}")
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is the number of heads, hs is the head size, and C (number of channels) = nh * hs
        # e.g in GPT2 (124M), n_head=12, hs=64, so nh*hs = 768 channels in the transformer
        
        logger.debug(f"causal self attention (forward section): B={B}, T={T}, C={C}")

        # qkv is the query, key, value projections
        # qkv is a tensor of shape (B, T, 3 * C)
        qkv = self.c_attn(x)
        logger.debug(f"causal self attention (forward section): qkv size - tensor of shape (B, T, 3 * C)")
        logger.debug(f"causal self attention (forward section): qkv size: {qkv.size()}")
        q, k, v = qkv.split(self.n_embd, dim=2)
        logger.debug(f"causal self attention (forward section): q size: {q.size()}, k size: {k.size()}, v size: {v.size()}")
        # q, k, v are tensors of shape (B, T, C)
        # TODO: (B, nh, T, hs) is the shape of q, k, v after the view and transpose operations below
        # B, nh -> batch size, number of heads, this is treated as batch size together.
        # Operations performed for all batches and all heads in parallel.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        logger.debug(f"causal self attention (forward section) after view and transpose, operations performed for all batches and all heads in parallel")
        logger.debug(f"causal self attention (forward section): q size: {q.size()}, k size: {k.size()}, v size: {v.size()}")

        # Following operation will be applied in parallel to all batches and all heads above
        # attention calculation (QK^T)
        logger.debug(f"At this point, at each position, for each token we have a key and a query. All done in parallel. None is yet communicating with each other. Lets comminicate")
        logger.debug(f"across batch dimension, we are not communicating, we are communicating only within the batch dimension")
        logger.debug(f"in the attention formula, we also have to divide by sqrt(d_k). d_k is the head size, also called scaled attention")
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        logger.debug(f"causal self attention (forward section): attention calculation (QK^T)")
        logger.debug(f"for every row in B, the effinities are given by a square matrix T x T")
        logger.debug(f"causal self attention (forward section): att size: {att.size()}")
        logger.debug(f"causal self attention (forward section): take the first row of batch, first head, 32 tokens, each token respresented with 32 size")
        logger.debug(f"causal self attention (forward section): att: {att[0, 0, :, :]}")
        # mask out the lower half of the dot product matrix, ensuring that q can only attend to k up to the current position
        # no looking into the future
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        logger.debug(f"causal self attention (forward section): att masked fill")
        logger.debug(f"causal self attention (forward section): att: {att[0, 0, :, :]}")
        # softmax - normalizing the attention weights, sums up to 1 each row
        att = F.softmax(att, dim=-1)
        logger.debug(f"causal self attention (forward section): att softmax")
        logger.debug(f"attention matrix below is for first row in batch, first head, 32 tokens, each token respresented with 32 size, the first row here shows 1.000 which means that its effinity is 1 with itself, and 0 with all other tokens, this is because of the masking, we are not allowing the token to look into the future, the token can only look at itself and the tokens before it, not after it, if you look at second row you will see that the token can look at itself and the token before it, if you now go to last row, you will see that the token can look at itself and all the tokens before it, the attention scores will clearly tell how much the effinities are between the tokens")
        logger.debug(f"causal self attention (forward section): att: {att[0, 0, :, :]}")
        # attention * values -> basically a weighted sum of the values, using the attention weights
        logger.debug(f"1. at the end we dont aggregate with x exactly, but with value v. 2. x is the private information of the token, v is more like a public information of the token 3. i am 5th token, my original identity is kept in x,  4. v instead has, for a single head, this is what i am interested in, here is what i have, if u find me inetresting here is what i will communicate to you")
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        logger.debug(f"causal self attention (forward section): y size (att @ v): {y.size()}")
        logger.debug(f"causal self attention (forward section): y: {y[0, 0, :, :]}")
        # reassamble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        logger.debug(f"causal self attention (forward section): y size (reassamble all head outputs): {y.size()}")
        logger.debug(f"causal self attention (forward section): y: {y[0, :, :]}")
        # final projection
        y = self.c_proj(y)
        logger.debug(f"causal self attention (forward section): y size (final projection): {y.size()}")
        logger.debug(f"causal self attention (forward section): y: {y[0, :, :]}")
        
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
        self.c_proj.GPT_SCALE_UNIT = 1
        
    def forward(self, x):
        logger.debug(f"mlp (forward section): x size: {x.size()}")
        logger.debug(f"mlp (forward section): MLP is a simple feed forward network with GELU activation")
        logger.debug(f"mlp (forward section): GELU is a smooth version of ReLU, its tail end is not straight line like ReLU, but a curve, this helps in better gradient flow and learning.")
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x    

class Block(nn.Module):
    # Block is always the combination of self attention and MLP
    # along with layer normalization and residual connections
    # ***Remember that in normal transformer architecture, the layer normalization is
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
        logger.debug(f"block (forward section): Started... Block")
        logger.debug(f"block (forward section): x size: {x.size()}")
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        logger.debug(f"block (forward section): Ended... Block")
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
        # modulelist can be indexed like a list e.g. self.transformer.h.0 or self.transformer.h[0]
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        logger.debug(f"model initialized...")
        logger.debug(f"transformer: {self.transformer}")
        logger.debug(f"lm_head: {self.lm_head}")

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        logger.debug(f"weight sharing scheme applied...")

        # iterate all the above modules and initialize their weights
        # initialize the weights (use the code - https://github.com/openai/gpt-2/blob/master/src/model.py)
        self.apply(self._init_weights)

    # initialize the weights, taken from the original gpt2 model
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): # initialize linear layers
            logger.debug(f"initializing linear layer: {module}")
            std = 0.02
            # GPT_SCALE_UNIT is a custom attribute, added as a flag, if you see, it is added both to attn and mlp. These both have
            # the residual pathways who's std need to be scaled down. Note: 2* is because of the two residual pathways.
            if hasattr(module, 'GPT_SCALE_UNIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # scale by the number of layers (scale down the std)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding): # initialize embedding layers
            logger.debug(f"initializing embedding layer: {module}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        logger.debug(f"training model...")
        B, T = idx.size() # batch size, sequence length
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        logger.debug(f"pos: {pos}")
        pos_emb = self.transformer.wpe(pos) # position embeddings for the sequence length T (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings for the sequence length T (B, T, n_embd)
        
        logger.debug(f"pos_emb size: {pos_emb.size()}")
        logger.debug(f"tok_emb size: {tok_emb.size()}")
        logger.debug(f"pos_emb: {pos_emb}")
        logger.debug(f"tok_emb: {tok_emb}")
        
        x = tok_emb + pos_emb # combine the token and position embeddings
        logger.debug(f"x size (combined pos+tok): {x.size()}")
        logger.debug(f"x: {x}")

        logger.debug(f"forwarding the block of transformer...")
        # forward the block of transformer
        cnt = 0
        for block in self.transformer.h:
            if cnt == 0:
                logger.debug(f"block: {block}")
            else:
                logger.disabled = True
            cnt += 1
            x = block(x)
            #import sys; sys.exit(0)
        # forward the final layer normalization and the linear layer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # cross entropy loss doesnt like the 3 dim (B, T, vocab_size) logits, lets flatten them to (B*T, vocab_size)
            # also the target needs to be of shape (B*T), not (B, T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # cross entropy loss
        return logits, loss


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
# create a log file to log all the details of model training
# create a logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check if log file exists
import os
if os.path.exists('/Users/manpreet.singh/git/gpt/notebooks/train.log'):
    # Remove the older log file
    os.remove('/Users/manpreet.singh/git/gpt/notebooks/train.log')

# create a file handler and set the level to DEBUG
file_handler = logging.FileHandler('/Users/manpreet.singh/git/gpt/notebooks/train.log')
file_handler.setLevel(logging.DEBUG)

# create a formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(file_handler)

# -------------------------------------------- Tensorboard Initialization -------------------------------------------- #

import tensorboard
from torch.utils.tensorboard import SummaryWriter
# specify the log directory
log_dir = '/Users/manpreet.singh/git/gpt/logs'
# create a SummaryWriter object
writer = SummaryWriter(log_dir)

# --------------------------------------------  -------------------------------------------- #
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        logger.debug(f"initializing DataLoaderLite with B={B}, T={T}")

        # at init loads tokens from disk and store them in memory
        with open('/Users/manpreet.singh/git/gpt/dataset/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens) # (B, T)

        logger.debug(f"loaded {len(self.tokens)} tokens")
        logger.debug(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1] # (B, T)
        x = buf[:-1].view(B, T) # input
        y = buf[1:].view(B, T) # output
        self.current_position += B * T # move the position
        # if loading the next batch will go out of bounds, reset the position
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        logger.debug(f"next_batch: x={x.size()}, y={y.size()}")
        logger.debug(f"next_batch: x={x}, y={y}")        
        return x, y

# --------------------------------------------  -------------------------------------------- #
# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # for apple mps
    device = "mps"
print(f"using device: {device}...")
# device = "cpu" # for now, we will use CPU

logger.debug(f"using device: {device}...")

torch.manual_seed(1337) # set the seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) # set the seed for reproducibility on GPU

'''
# get a data batch
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('/Users/manpreet.singh/git/gpt/dataset/input.txt', 'r') as f:
    text = f.read()
text = text[:1000] # only the first 1000 characters
tokens = enc.encode(text)
B, T = 4, 32 # batch size, sequence length
buf = torch.tensor(tokens[:B*T + 1]) # (B, T)
# you cannot move buf in stateful manner to gpu, instead for a tensor you will get a pointer to the tensor placed on the GPU
buf = buf.to(device) # move the tensor to the GPU on Cloudbox
x = buf[:-1].view(B, T) # input
y = buf[1:].view(B, T) # output
'''

# get a data batch
train_loader = DataLoaderLite(B=4, T=32)

# get logits
model = GPT(GPTConfig()) # 124M
model.to(device) # move the model to the GPU on Cloudbox
# logits, loss = model(x, y) # (B, T, vocab_size)

logger.debug(f"model initialized...")
logger.debug(f"model: {model.config}")

# optimize
# you will see for single batch, loss starts with 10.5, and then decreases to 0.002. Perfectly overfitting the single batch :)
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
# FP32: 19.5 TFLOPS, FP16: 156 TFLOPS, TF32: 312 TFLOPS, meaning as the float precision decreases, the performance increases i.e. 
# the T FLOPS increases (trillion floating point operations a.k.a tera flops).
# Floating point operations are important for training as it allows the distribution of values to be what we want, wehereas INT
# operations are very fast like INT8 (TFLOPS = 624) and equally spaced, so are not used for training, they are used for inference.
# Smaller the number of bits to represent a number, easy it is to move the data around. Also faster.
# Multiple GPU tensor cores perform operations at great speed (tera flops), but they are contraint by the memory bandwidth (speed at which the the bits are read from the memory). Because the data needs to be moved to the tensor cores, and if the memory bandwidth is slow, the tensor cores will be idle. In nutshell, even 60% of usage of tensor cores is great.
# https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
# All our main operations are matrix multiplications, and tensor cores are great at matrix multiplications.
# All linear layers are matrix multiplications, where GELU, LayerNorm, softmax are not very deep operations. Also if you see the biggest matrix multiplication is the 768 (embedding size) to 50257(vocab size) conversion at the top.
# https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
# Let talk a little bit about tensor float in the paper above. Tensor float is a new floating point representation,
# when we run on the GPU, we run on the tensor cores, and tensor cores are designed to run on the tensor float format, meaning that
# matrix multiplications is of format a * b + c, where a, b, c are 4x4 matrices. This is the tensor float format. This is a new format
# where the floating point is not 32 bits but 23 bits e.g. [sign, 8 bit exponent (range), 14 bit mantissa(precision)], remaining bits from mantissa are
# removed. Yes it does reduce the precision, but it is good enough for training. The tensor cores are designed to run on this format, and they are very fast (8x faster).
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # AdamW optimizer (fixes a bug in Adam).
logger.debug(f"optimizer initialized...")
logger.debug(f"optimizer: {optimizer}")

for i in range(20):
    logger.debug(f"step: {i}")
    x, y = train_loader.next_batch() # get the next batch
    x, y = x.to(device), y.to(device) # move to GPU
    optimizer.zero_grad() # reset the gradients
    logits, loss = model(x, y) # forward pass

    # Convert logits to actual text
    enc = tiktoken.get_encoding('gpt2')
    predicted_tokens = torch.argmax(logits, dim=-1)
    decoded_text = enc.decode(predicted_tokens[0].cpu().numpy())
    
    writer.add_scalar('Loss/train', loss.item(), i)
    writer.add_histogram('Logits', logits, i)
    metadata = [f"token_{i}" for i in range(logits.size(0))]
    writer.add_embedding(predicted_tokens, metadata, global_step=i)
    writer.flush()
    # import code; code.interact(local=locals()) # drop into an interactive shell
    loss.backward() # backward pass
    optimizer.step() # update the weights
    print(f"step: {i}, loss: {loss.item()}") # loss is a scalar tensor, it is shipped from gpu to cpu (float), and printed
    

# remember that at initialization, the probability of each voc at the output should be uniform, i.e. 1/vocab_size. Remember that we dont
# want any vocabulary to be favored at the start, we want the model to learn the distribution of the vocabulary.
# refer: loss_at_init.png

print(loss) # 
import sys; sys.exit(0)

num_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('gpt2') # 124M 
model = GPT(GPTConfig()) # 124M
model.eval() # set to eval mode, model is not in training, we will just be using it for inference
model.to(device) # move the model to the GPU on Cloudbox

# prefix tokens
import tiktoken
import os
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # convert to tensor (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # repeat for number of sequences (5, 8) (B, T)
x = tokens.to(device) # move to GPU

# generate! right now x is (B, T) where B is the number of sequences to generate (5, 8)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
print(x.size())
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

        

