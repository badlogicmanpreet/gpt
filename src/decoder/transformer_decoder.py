import torch
import torch.nn as nn
from torch.nn import functional as F

# Transformer decoder - by Manpreet Singh

# hyperparameters
batch_size = 128 # number of sequences in a batch
block_size = 512 # length of each sequence
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # embedding size
n_head = 6 # number of heads in the multi-head attention
n_layer = 6 # number of transformer blocks
dropout = 0.2

torch.manual_seed(1337)

# Read the input file and print the length of the text
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create the character to index and index to character mapping - tokenization
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

# lets split into training and validation set
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% training, 10% validation
train_data, valid_data = data[:n], data[n:]

# data loader
def get_batch(split):
    """
    Get a batch of data from the specified split.

    Args:
        split (str): The split to get the batch from. Can be 'train' or 'val'.

    Returns:
        tuple: A tuple containing the input sequences and target sequences.
    """
    if split == 'train':
        data = train_data
    else:
        data = valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # starting index of each sequence
    x = torch.stack([data[i:i+block_size] for i in ix]) # get the sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # get the target sequences
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss on the training and validation sets.

    Returns:
        dict: A dictionary containing the average loss on the training and validation sets.
    """
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

class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        # linear projections for our nodes
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Perform self-attention on the input sequence.

        Args:
            x (torch.Tensor): The input sequence of shape (batch_size, block_size, n_embd).

        Returns:
            torch.Tensor: The output sequence after self-attention of shape (batch_size, block_size, n_embd).
        """
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores, effinities
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) # projection layer going back into the residual pathway
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Perform multi-head self-attention on the input sequence.

        Args:
            x (torch.Tensor): The input sequence of shape (batch_size, block_size, n_embd).

        Returns:
            torch.Tensor: The output sequence after multi-head self-attention of shape (batch_size, block_size, n_embd).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out 
        
class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
        
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 4 comes from the paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # going back into the residual pathway
            nn.Dropout(dropout), # dropout layer
        )
        
    def forward(self, x):
        """
        Perform feed-forward computation on the input sequence.

        Args:
            x (torch.Tensor): The input sequence of shape (batch_size, block_size, n_embd).

        Returns:
            torch.Tensor: The output sequence after feed-forward computation of shape (batch_size, block_size, n_embd).
        """
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # 4 heads of 8 dimensional self attention (here each head (communication channel) will give us 8 dimensions, so 4*8 = 32, which is the emb_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        """
        Perform communication and computation on the input sequence.

        Args:
            x (torch.Tensor): The input sequence of shape (batch_size, block_size, n_embd).

        Returns:
            torch.Tensor: The output sequence after communication and computation of shape (batch_size, block_size, n_embd).
        """
        x = x + self.sa(self.ln1(x)) # apply one head of self-attention
        x = x + self.ffwd(self.ln2(x)) # apply feed forward layer
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding layer
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # positional encoding, spatial information is important
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # 4 transformer blocks
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer
        
    def forward(self, idx, targets=None):
        """
        Perform forward pass of the GPT model.

        Args:
            idx (torch.Tensor): The input sequence of shape (batch_size, block_size).
            targets (torch.Tensor): The target sequence of shape (batch_size, block_size).

        Returns:
            tuple: A tuple containing the logits and loss (if targets are provided).
        """
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx) # (batch_size, block_size, emb_size) (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (block_size, emb_size) (T, C)
        x = token_emb + pos_emb # (batch_size, block_size, emb_size) (B, T, C)

        x = self.blocks(x) # apply the transformer blocks (B, T, C)
        x = self.ln_f(x) # apply final layer norm (B, T, C)
        logits = self.lm_head(x) # (batch_size, block_size, vocab_size) (B, T, vocab_size) e.g. (1, 32, 65), for 1 token batch size, 32 is the input and 65 is the output of linear layer

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens given an input sequence.

        Args:
            idx (torch.Tensor): The input sequence of shape (batch_size, block_size).
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            torch.Tensor: The generated sequence of shape (batch_size, block_size + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond) # (B*T, C)
            logits = logits[:, -1, :] # focus on the last time step only
            probs = F.softmax(logits, dim=-1) # apply softmax to get the probabilities
            next_token = torch.multinomial(probs, num_samples=1) # sample from the distribution
            idx = torch.cat([idx, next_token], dim=1) # append sampled index to the running sequence
        return idx

model = GPT()
m = model.to(device)

# create a optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, valid loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))





