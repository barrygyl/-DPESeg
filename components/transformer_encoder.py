# 人工智能
# 项目：segment
# 开发人：高云龙
# 开发时间：2023-07-11  14:19
# 开发工具：PyCharm

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from components.MLP import Mlp
from components.self_Attention import Attention

"""transformer编码层"""


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights






