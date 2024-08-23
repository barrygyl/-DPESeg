# 项目： segment
# 开发人：高云龙
# 开发时间：  15:38
# 开发工具： PyCharm
import torch
import torch.nn as nn
from components.self_Attention import Attention
from components.MLP import Mlp
from components.Attention import Attention as img_token_att



class PromptDecoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embed = nn.Embedding(configs.vocab_size, embedding_dim=configs.embed_dim)
        self.self_att = Attention(configs ,vis=True)
        self.norm = nn.LayerNorm(configs.embed_dim)
        self.norm1 = nn.LayerNorm(configs.embed_dim)
        self.norm2 = nn.LayerNorm(configs.embed_dim)
        self.norm3 = nn.LayerNorm(configs.embed_dim)
        self.norm4 = nn.LayerNorm(configs.embed_dim)
        self.mlp = Mlp(configs)
        self.i_t_att = img_token_att(embedding_dim=configs.embed_dim, num_heads=configs.num_heads)
        self.t_i_att = img_token_att(embedding_dim=configs.embed_dim, num_heads=configs.num_heads)
        self.fin_att = img_token_att(embedding_dim=configs.embed_dim, num_heads=configs.num_heads)

    def forward(self, x, p):
        """先验知识融合"""
        t_f = self.embed(p)
        t_f = self.self_att(t_f)[0] + t_f
        t_f = self.norm(t_f)
        t_f = self.i_t_att(t_f, x, x) + t_f
        t_f = self.norm1(t_f)
        t_f = self.mlp(t_f) + t_f
        t_f = self.norm2(t_f)
        out = self.t_i_att(x, t_f, t_f) + x
        t_f = self.fin_att(t_f, out, out) + t_f
        t_f = self.norm4(t_f)
        return out, t_f
