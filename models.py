# 人工智能
# 项目：segment
# 开发人：GYL
# 开发时间：2023-05-31  20:32
# 开发工具：PyCharm
"""Dynamic Multi-Organ Segmentation Model Incorporating Prior Features"""
import copy

import torch
from torch import nn
from torch.nn import LayerNorm


import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from components.Res_conv import Mul_conv
from components.Img_Embed import Embeddings
from components.transformer_encoder import Block
from components.Decoder import Decoder, Decoder_pre, SegmentationHead
from components.Prompt_Decoder import PromptDecoder


'''模型的构建部分'''

__all__ = ['UNet', 'NestedUNet', 'DM_net', 'AttUnetRet']  # 构建的模型名字放在里面


class DM_net(nn.Module):
    def __init__(self, configs, input_channels=3, **kwargs):
        super().__init__()

        self.mul_conv = Mul_conv(configs.block_units, configs.width_factor)
        # self.cwpm = CWPM(configs.pre_size, int(configs.pre_size / 2), int(configs.pre_size / 4), configs.switch_size)
        self.embed = Embeddings(configs, in_channels=self.mul_conv.width * 16)
        self.encoder_layers = nn.ModuleList()
        for _ in range(configs.trans_encoder_nums):
            layer = Block(configs, vis=True)
            self.encoder_layers.append(copy.deepcopy(layer))
        self.encoder_norm = LayerNorm(configs.hidden_size, eps=1e-6)
        self.decoder = Decoder(configs)
        self.decoder_pre = Decoder_pre(configs)
        self.segment_head = SegmentationHead(configs)

        self.relu = nn.ReLU()
        self.pro_decoder = PromptDecoder(configs)
        self.pooling_layer = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x, t_f=None):
        """encoder编码"""
        '''图像特征抽取转embedding'''
        x, features = self.mul_conv(x)

        x = self.embed(x)
        # attentions_weights = [0, 0, 0, 0, 0]  # 注意力模块的权重
        for i, layer_block in enumerate(self.encoder_layers):
            x, a_w = layer_block(x)
            # attentions_weights[i] = a_w
        
        x = self.encoder_norm(x)
        """先验知识融合"""
        x, t_f = self.pro_decoder(x, t_f)  # t_f(8,1,768)
        t_f = t_f.unsqueeze(3).permute(0, 2, 1, 3)
        t_f = torch.sigmoid(t_f)
        encoder = self.relu(x)
        for_decoder = self.decoder_pre(encoder)
        t_f = self.pooling_layer(t_f)
        decoder = self.decoder(for_decoder, features, t_f)
        output = self.segment_head(decoder)
        return output

