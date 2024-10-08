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
import inspect

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

        # NORMAL ATTENTION
        # Following operation will be applied in parallel to all batches and all heads above
        # attention calculation (QK^T)
        '''
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
        '''
        
        # In normal attention, the python interpreter does things one by one, uses GPU/HBM back and forth.
        # We will use flash attention, so that torch.compile can take advantage of kernel fusion, and the entire attention operation can be done in one go.
        # FLASH ATTENTION - https://arxiv.org/pdf/2205.14135, FLASH ATTENTION2 - https://arxiv.org/pdf/2307.08691
        # Flash attention actuall does more FLOPs, but due to kernel fusion, it is faster. 7.6x faster than normal attention.
        # Flash attention is extremely careful about the memory hierarchy.
        # The att matrix on line 62, is actually never stored or read completely, it never gets materialized, no LOAD/STORE operations from HBM. 
        # This is only possible becuase of online softmax trick. In here you can incremently evaluate the softmax, you dont need to have the entire matrix in memory.
        # We use intermiddiate variables called m and l to calculate the softmax. The softmax is calculated incremently. https://arxiv.org/pdf/1805.02867
        # FLOPS dont matter, the memory hierarchy matters.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)


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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the parameters in the model and separate out all parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim group, any param that is 2D will decay else no
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorm params do not
        # weight decay will pull down the weights, so that the model does not overfit, it will make the weights smaller
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decay parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num no decay parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create the AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using {'fused' if use_fused else 'unfused'} AdamW optimizer")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

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
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

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
        self.current_position = self.B * self.T * self.process_rank # start at the beginning of the process's data

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1] # (B, T)
        x = buf[:-1].view(B, T) # input
        y = buf[1:].view(B, T) # output
        self.current_position += B * T * self.num_processes # move the position
        # if loading the next batch will go out of bounds, reset the position
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        logger.debug(f"next_batch: x={x.size()}, y={y.size()}")
        logger.debug(f"next_batch: x={x}, y={y}")        
        return x, y

# --------------------------------------------  -------------------------------------------- #
import time

# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # for apple mps
    device = "mps"
print(f"using device: {device}...")
# device = "cpu" # for now, we will use CPU

logger.debug(f"using device: {device}...")

#--------------------------------------------  -------------------------------------------- #
# simple run
# python train_gpt_with_experiments.py
# DDP run, with 8 GPUs
# torchrun --standalone --nproc_per_node=8 train_gpt_with_experiments.py

# distributed the model to 8 GPUs

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# setup distributed data parallel training
# torchrun command sets the env variables, RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # distributed data parallel
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to the RANK
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl') # initialize the process group
    ddp_rank = int(os.environ['RANK']) # global rank, gpu number for the script
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # single node with multiple GPUs, local rank is the GPU number
    ddp_world_size = int(os.environ['WORLD_SIZE']) # world size (number of processes) 8 in this case
    device = f"cuda:{ddp_local_rank}" # which process is running on which GPU, cude:0, cuda:1, etc.
    torch.cuda.set_device(device) # set the device
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, etc. (we set this, other processes are more like compute processes)
else:
    # vanilla non distributed training
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # automatically detect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # for apple mps
        device = "mps"
    print(f"using device: {device}...")

# --------------------------------------------  -------------------------------------------- #
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

# we need to work on a batch size like 0.5M tokens, so that we can train the model in a reasonable time (like gpt3 small)
# but we dont have that much memory where .5M batch size can be taken, so we will need to work with streaming data/batch, called gradient accumulation
# we will need to accumulate the gradients for multiple batches, and then update the weights, this is called gradient accumulation.
# Basically what we do here is, we keep going forward backward and keep doing += on the gradients, and then after a certain number of steps (gradient accumulation steps),
# we do the update/optimizer step.
total_batch_size = 524288 # 2**19, 0.5M tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "total batch size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # number of steps to accumulate gradients over, basically do 32 forward backward passes and then do the optimizer step
if master_process: # only log this on the master process (gpu0)
    print(f"total desired batch size: {total_batch_size:}")
    print(f"==> calculated gradient accumulation steps: {grad_accum_steps}")

# print("I am gpu ", ddp_rank)
# import sys ; sys.exit(0)

# get a data batch, we need to make data loader lite aware of the ddp processes, so that it can load the data accordingly
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, world_size=ddp_world_size)

# set the float32 matmul precision to high, i have A100 GPU, so i can use the high precision (Ampere Series GPU).
# total options available are highest, high, medium
# Also remember it says that throughput will increase by 8x, but since everywhere else float32 is being shipped(memory bound) and dealth with, its not 8x, but 1.2x.
# lets now use BFLOAT16, range is same, mantisa is further compressed, so the precision is further reduced, but it is good enough for training.
# use the pytorch document, to read more about Automatic mixed precision training. Look only at torch.autocast. Follow the guidance
torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304)) # 124M (50304 is the padded vocab size, beautiful number)
model.to(device) # move the model to the GPU on Cloudbox

# compile the model, read the pytorch documentation, it is a good practice to compile the model, it will optimize the model for the given device.
# compile will remove the python interpreter overhead, it will look at the forward method in one shot and compile it.
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # distributed data parallel
    # so every local rank (gpu) will have a copy of the model, and they will communicate with each other, the forward pass goes through simply
    # and during the backward pass, the gradients are communicated across the GPUs to get average and then each rank holds the average gradient (all reduce operation),
    # and the optimizer step is done on each GPU.
raw_model = model.module if ddp else model # if DDP, we need to access the underlying model to configure the optimizer

# logits, loss = model(x, y) # (B, T, vocab_size)

logger.debug(f"model initialized...")
logger.debug(f"model: {model.config}")

max_lr = 6e-4 # refer the doc
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iter, return minimum learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to minimum learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cosine decay
    return min_lr + coeff * (max_lr - min_lr)

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
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # AdamW optimizer (fixes a bug in Adam).

# The goal is to have a batch size 0.5M tokens, but we dont have that much memory, so we will use gradient accumulation.
# We will accumulate the gradients(+=) for multiple batches, and then do the optimizer step.
# Basically what we do here is, we keep going forward backward and keep doing += on the gradients, and then after a certain number of steps (gradient accumulation steps),
# we loop through max_steps (50), and in each step we do the following:
# 1) zero the gradients
# 2) for each microstep, get the next batch
#    a) move the batch to the GPU
#    b) forward pass
#    c) scale the loss (1/4 example)
#    d) accumulate the loss
#    e) backward pass
# 3) clip the global norm of the gradients to 1.0
# 4) update the learning rate
# 5) update the weights
# 6) print the loss, learning rate, norm, time taken, tokens per second

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device) # configure the optimizer
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() # reset the gradients
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # get the next batch
        x, y = x.to(device), y.to(device) # move to GPU    
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # use bfloat16 precision
            logits, loss = model(x, y) # forward pass
        loss = loss / grad_accum_steps # scale the loss
        # if ddp, we want to not only get loss on master rank, but also on other ranks, so we need to all reduce the loss
        loss_accum += loss.detach() # accumulate the loss
        # all reduce operation should only happen at the last step of the gradient accumulation
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # turn on gradient sync only when micro_step is the last step
        loss.backward() # backward pass (accumulate the gradients with +=)
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # all reduce the loss
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step() # update the weights
    torch.cuda.synchronize() # wait for everything to finish, wait for the GPU to finish all the above before the cpu executes the next line
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size # number of tokens processed
    tokens_per_sec =  tokens_processed / dt
    if master_process:
        print(f"step: {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}") # loss is a scalar tensor, it is shipped from gpu to cpu (float), and printed
    
# destroy the process group
if ddp:
    destroy_process_group()

'''
for i in range(50):
    t0 = time.time()
    #logger.debug(f"step: {i}")
    x, y = train_loader.next_batch() # get the next batch
    x, y = x.to(device), y.to(device) # move to GPU
    optimizer.zero_grad() # reset the gradients

    with torch.autocast(device_type=device, dtype=torch.bfloat16): # use bfloat16 precision
        logits, loss = model(x, y) # forward pass
        #import code; code.interact(local=locals()) # drop into an interactive shell

    # Calculate the total number of trainable parameters
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.debug(f"Total number of trainable parameters: {total_params}")
    # print(f"Total number of trainable parameters: {total_params}")

    # Convert logits to actual text
    # enc = tiktoken.get_encoding('gpt2')
    # predicted_tokens = torch.argmax(logits, dim=-1)
    # decoded_text = enc.decode(predicted_tokens[0].cpu().numpy())
    
    #writer.add_scalar('Loss/train', loss.item(), i)
    #writer.add_histogram('Logits', logits, i)
    #writer.flush()
    # import code; code.interact(local=locals()) # drop into an interactive shell
    loss.backward() # backward pass
    # clip the global norm of the gradients to 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step() # update the weights
    torch.cuda.synchronize() # wait for everything to finish, wait for the GPU to finish all the above before the cpu executes the next line
    t1 = time.time()
    dt = (t1 - t0)*1000 # in milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step: {i}, loss: {loss.item()}, norm: {norm:.4f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}") # loss is a scalar tensor, it is shipped from gpu to cpu (float), and printed
'''

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

        

