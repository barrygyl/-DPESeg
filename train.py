# 人工智能
# 项目：
# 开发人：Barry
# 开发时间：2023-05-31  20:32
# 开发工具：PyCharm

import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 这里输入你的GPU_id
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 这里输入你的GPU_id
from collections import OrderedDict

import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn

import models
import losses
from metrics import iou_score, dice_coef
from utils import AverageMeter, custom_init

# 禁止显示警告
warnings.filterwarnings('ignore')
# 调整占用的GPU卡

torch.cuda.empty_cache()

# 通用变量
ARCH_NAMES = models.__all__
LOSS_NAMES = losses.__all__
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_NAMES.append('BCEWithLogitsLoss')

"""

指定参数：
--dataset dsb2018_96 
--arch NestedUNet
--loss BCEDiceLoss
"""


# 设置初始参数


def train(config, train_loader, model, criterion, optimizer, Device):
    # AverageMeter计算和存储平均值和当前值。
    avg_meters = {'loss': AverageMeter(), 
                  'Dice': AverageMeter(), 

                 }

    model.train()

    pbar = tqdm(total=len(train_loader))
    # 加载数据集
    for input, target, text_idx, organ_features, _ in train_loader:

        input = input.cuda()
        target = target.cuda()
        text_idx = text_idx.cuda()

        # compute output
        output = model(input, t_f=text_idx)
        outputs = []
        for i in range(output.shape[0]):
            a = int(organ_features[i])
            outputs.append(output[i, a, :, :])
        output = torch.stack(outputs)
        loss = criterion(output.unsqueeze(dim=1), target.unsqueeze(dim=1))
        dice = dice_coef(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['Dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('Dice', avg_meters['Dice'].avg),
        ])
        pbar.set_postfix(postfix)  # 这一行代码设置了进度条的后缀。
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('Dice',avg_meters['Dice'].avg)])


def validate(config, val_loader, model, criterion, Device):
    avg_meters = {'loss': AverageMeter()
                  , 'Dice': AverageMeter()
                  }

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, text_idx, organ_features, _ in val_loader:
            input = input.to(Device)
            target = target.to(Device)
            text_idx = text_idx.to(Device)

            # compute output

            output = model(input, t_f=text_idx)
            outputs = []
            for i in range(output.shape[0]):
                a = int(organ_features[i])
                outputs.append(output[i, a, :, :])
            output = torch.stack(outputs)
            loss = criterion(output.unsqueeze(dim=1), target.unsqueeze(dim=1))
            dice = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['Dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('Dice', avg_meters['Dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('Dice',avg_meters['Dice'].avg)])


def main(config, train_loader, val_loader, model, criterion, optimizer, scheduler, Device):
    # 加载上次的训练模型参数
    # if os.path.exists('/data/GYL/Models_info/saved_models/%s/model.pth' % config['name']):
    #     model.load_state_dict(torch.load('/data/GYL/Models_info/saved_models/%s/model.pth' %
    #                                     config['name']))
    model_dict = model.state_dict()
    pretrained_dict = torch.load('/data/GYL/Models_info/saved_models/DM_net/PAST/model_21_transUnet.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for name, param in model.named_parameters():
        if name in pretrained_dict:
            # param.requires_grad = False
            pass
        else:
            try:
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                if 'bias' in name:
                    nn.init.zeros_(param)
            except:
                pass
    model.train()
    # 设置训练变量存储字典
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('Dice', []),
        ('val_loss', []),
        ('val_Dice', []),
    ])

    best_dice = 0  # 最佳交并比（IoU）的变量
    trigger = 0  # 触发器变量
    print('-' * 50)
    print('\n')

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, Device)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion, Device)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - Dice %.4f - val_loss %.4f - val_Dice %.4f'
              % (train_log['loss'], train_log['Dice'], val_log['loss'], val_log['Dice']))

        log['epoch'].append(epoch)
        log['lr'].append(scheduler.get_last_lr()[0])
        log['loss'].append(train_log['loss'])
        log['Dice'].append(train_log['Dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_Dice'].append(val_log['Dice'])

        pd.DataFrame(log).to_csv('/data/GYL/Models_info/saved_models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['Dice'] > best_dice:
            torch.save(model.state_dict(), '/data/GYL/Models_info/saved_models/%s/model.pth' %
                       config['name'])
            best_dice = val_log['Dice']
            print("=> saved best model <=")
            trigger = 0

        # early stopping
        if 0 <= config['early_stopping'] <= trigger:
            print("=> early stopping <=")
            break

        torch.cuda.empty_cache()
        print('\n')
        print('-' * 50)
        print('\n')


