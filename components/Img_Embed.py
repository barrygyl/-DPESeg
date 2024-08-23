# 人工智能
# 项目：segment
# 开发人：高云龙
# 开发时间：2023-07-11  15:22
# 开发工具：PyCharm

import torch
import torch.nn as nn
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair

'''图像转Embedding'''


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, in_channels=1024):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(config.img_size)

        if config.patches["grid"] is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.trans_dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


if __name__ == "__main__":
    from Config import Config
    x = Config({'arch':'da'})
    model = Embeddings(x)