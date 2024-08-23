# 人工智能
# 项目：segment
# 开发人：高云龙
# 开发时间：2023-07-12  9:58
# 开发工具：PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from components.VGGBlock import VGGBlock
from components.Attention import SptailAttention

'''解码前的处理模块'''


class Decoder_pre(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_more = VGGBlock(in_channels=config.hidden_size,
                                  middle_channels=config.decoder_pre_out,
                                  out_channels=config.decoder_pre_out)

    def forward(self, x):
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        return x


'''skip-解码'''


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = VGGBlock(in_channels=config.decoder_pre_out * 2,
                              middle_channels=int(config.decoder_pre_out),
                              out_channels=int(config.decoder_pre_out / 2))
        self.conv2 = VGGBlock(in_channels=int(config.decoder_pre_out),
                              middle_channels=int(config.decoder_pre_out / 2),
                              out_channels=int(config.decoder_pre_out / 8))
        self.conv3 = VGGBlock(in_channels=int(config.decoder_pre_out / 4),
                              middle_channels=int(config.decoder_pre_out / 8),
                              out_channels=int(config.decoder_pre_out / 8))
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.att1 = SptailAttention(config.hidden_size, config.decoder_pre_out)
        self.att2 = SptailAttention(config.hidden_size, config.decoder_channels[0])
        self.att3 = SptailAttention(config.hidden_size, config.decoder_channels[2])

    def forward(self, x, feature, t):
        if t is not None:
            x = self.att1(x, t) + x
            x = self.up(x)
            x = torch.cat([x, feature[0]], dim=1)
            x = self.conv1(x)
            x = self.att2(x, t) + x 
            x = self.up(x)
            x = torch.cat([x, feature[1]], dim=1)
            x = self.conv2(x)
            x = self.att3(x, t) + x
            x = self.up(x)
            x = torch.cat([x, feature[2]], dim=1)
            x = self.conv3(x)
        else:
            x = self.up(x)
            x = torch.cat([x, feature[0]], dim=1)
            x = self.conv1(x)
            x = self.up(x)
            x = torch.cat([x, feature[1]], dim=1)
            x = self.conv2(x)
            x = self.up(x)
            x = torch.cat([x, feature[2]], dim=1)
            x = self.conv3(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, config, kernel=3, upsampling=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=config.decoder_channels[-1], out_channels=config.class_num,
                              kernel_size=kernel, padding=kernel // 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        # x = F.softmax(x)
        return x
