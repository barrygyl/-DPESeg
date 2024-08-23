
class Config(object):
    """配置参数"""

    def __init__(self, config):
        self.pre_size = 32
        self.switch_size = 5
        self.model_name = config['arch']
        self.hidden_size = 768  # 隐藏特征大小
        self.img_size = (192, 192)  # 图像大小
        self.patches = {'size': (1, 1), 'grid': (12, 12)}  # emb分割大小
        self.block_units = (3, 4, 9)  # 每个块（block）中单元（unit）数量的列表或元组。
        self.width_factor = 1  # width_factor 是一个缩放因子，用于控制模型的宽度。
        self.trans_encoder_nums = 12  # transformer_encoder的数量
        self.class_num = 21  
        self.decoder_pre_out = 512  # 解码前的最终通道数
        self.decoder_channels = [256, 126, 64]
        self.data_path = '/data/GYL/HNS1_DATA/PET'  # 数据集位置
        self.trans_dropout_rate = 0.5
        self.trans_mlp_dim = 3072  # transformer_encoder的全连接层
        self.num_heads = 24  # 多头注意力机制头数
        self.vocab_size = 30522
        self.embed_dim = 768
        self.deep_supervision = False

        self.u_class_nums = 21
