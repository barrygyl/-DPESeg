# 人工智能
# 项目：segment
# 开发人：高云龙
# 开发时间：2023-07-11  15:19
# 开发工具：PyCharm
import torch
import torch.nn as nn
from torch.nn import Dropout, Linear

'''全连接层'''


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.trans_mlp_dim)
        self.fc2 = Linear(config.trans_mlp_dim, config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.trans_dropout_rate)

        self._init_weights()  # 初始化系数

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x