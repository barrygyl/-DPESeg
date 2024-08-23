# 项目： segment
# 开发人：高云龙
# 开发时间：  14:02
# 开发工具： PyCharm
import warnings
import argparse

import models
import losses
from utils import str2bool
# 禁止显示警告
warnings.filterwarnings('ignore')

"""用于获取配置参数"""

# 通用变量
ARCH_NAMES = models.__all__
LOSS_NAMES = losses.__all__


def parse_args():
    parser = argparse.ArgumentParser()
    """基本参数进入"""
    # 模型 DM_net UNet AttUnetRet NestedUNet
    parser.add_argument('--name', default='DM_net',
                        help='model name: (default: arch+timestamp)')
    # 输入的通道数
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # 分类数
    parser.add_argument('--num_classes', default=21, type=int,
                        help='number of classes')
    
    """数据集参数"""
    # dataset
    parser.add_argument('--dataset', default='PET',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    

    """调参部分"""
    # loss损失函数选择  BCEDiceLoss
    parser.add_argument('--loss', default='DiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: LovaszHingeLoss)')
    # 训练轮次
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练批次大小
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        metavar='N', help='mini-batch size')

    # 模型选择
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DM_net',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: DM_net)')
    # 优化器
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    # 学习率
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    '''学习率调整'''
    # 学习率调度器
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    # 学习率最小值
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    # 余弦退火的幅度因子
    parser.add_argument('--factor', default=0.1, type=float)
    # 触发机制
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)

    # 早停
    parser.add_argument('--early_stopping', default=100, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config