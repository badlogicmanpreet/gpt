# GPT2 is a decoder only model, encoder and the cross attention piece is completely removed - https://arxiv.org/pdf/1706.03762
# In GPT2, https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
  #Layer normalization (Ba et al., 2016)
  #was moved to the input of each sub-block, similar to a
  #pre-activation residual network (He et al., 2016) and an
  #additional layer normalization was added after the final selfattention block. 

from dataclasses import dataclass
import torch
import torch.nn as nn
