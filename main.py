# 项目： segment
# 开发人：高云龙
# 开发时间：  16:55
# 开发工具： PyCharm

import os
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from glob import glob
import albumentations as alb
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
import warnings
import pickle as pkl

import losses
import models
from parser_config import parse_args
from Config import Config
from dataset import Dataset
from train import main as train
from val import main as test

"""主函数"""
# 禁止显示警告
warnings.filterwarnings('ignore')
# 调整占用的GPU卡
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 这里输入你的GPU_id
torch.cuda.empty_cache()

# 设备
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Device = torch.device('cpu')

def get_subdirectories(folder_path):
    subdirectories = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subdirectories.append(item)
    return subdirectories

def main():
    config = vars(parse_args())
    os.makedirs('/data/GYL/Models_info/saved_models/%s' % config['name'], exist_ok=True)

    print('-' * 50)
    print('基本配置信息：')

    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 50)

    # 将基本配置保存
    with open('/data/GYL/Models_info/saved_models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        criterion = losses.__dict__[config['loss']](config).to(Device)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s <=" % config['arch'])
    config_t = Config(config)
    model = models.__dict__[config['arch']](config_t)

    model = model.to(Device)

    # 筛选模型中需要训练的参数
    params = filter(lambda p: p.requires_grad, model.parameters())
    # 优化器
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # 调整合适地学习率调度器
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])  # config['epochs']
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    with open('/home/gyl/project/segment/train_s.pkl', 'rb') as f:
        train_img_ids = pkl.load(f)
    with open('/home/gyl/project/segment/test_s.pkl', 'rb') as f:
        val_img_ids = pkl.load(f)

    # 数据增强：
    train_transform = Compose([
        alb.RandomRotate90(),  # 随机角度反转
        alb.Flip(),  # 图像翻转
        OneOf([
            transforms.HueSaturationValue(),  # 色调饱和度值
            transforms.RandomBrightness(),  # 随机亮度
            transforms.RandomContrast(),  # 随机对比度
        ], p=1),  # 按照归一化的概率选择执行哪一个
        transforms.Normalize(),  # 标准化
    ])

    val_transform = Compose([
        transforms.Normalize(),
    ])  # 验证集用原始图像做

    # 设置训练集
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config_t.data_path, 'images', 'separate'),
        mask_dir=os.path.join(config_t.data_path, 'n_mask_21'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config_t.class_num,
        transform=train_transform)

    # 设置验证集
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config_t.data_path, 'images', 'separate'),
        mask_dir=os.path.join(config_t.data_path, 'n_mask_21'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config_t.class_num,
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    train(config, train_loader, val_loader, model, criterion, optimizer, scheduler, Device)
    print('开始测试！')
    test(val_loader, model)



if __name__ == '__main__':
    print('\n')
    print('开始训练！')
    main()
