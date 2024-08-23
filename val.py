import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as alb
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import segmentation_models_pytorch as smp
from parser_config import parse_args
import pandas as pd 

from metrics import iou_score, dice_coef, get_stats, HD
from utils import AverageMeter

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
测试 DM_net
"""


# def parse_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--name', default='UNet',
#                         help='model name')

#     args = parser.parse_args()

#     return args


def main(val_loader, model):
    columns = ['Jaccard', 'HD', 'Dice', 'Recall']


    # 创建一个空的DataFrame，列的值初始化为NaN  
    ans = pd.DataFrame(columns=columns)  
    jaccard = []
    HDlist = []
    Dice = []
    recall = []
    args = parse_args()

    with open('/data/GYL/Models_info/saved_models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 50)
    cudnn.benchmark = True

    # loading model
    print("=> loading model %s to eval<=" % config['arch'])

    model = model.to(Device)
    if os.path.exists('/data/GYL/Models_info/saved_models/%s/model.pth' % config['name']):
        model.load_state_dict(torch.load('/data/GYL/Models_info/saved_models/%s/model.pth' %
                                        config['name']))
    # if os.path.exists('/data/GYL/Models_info/saved_models/%s/PAST/model_fin_21.pth' % config['name']):
    #     model.load_state_dict(torch.load('/data/GYL/Models_info/saved_models/%s/PAST/model_fin_21.pth' %
    #                                     config['name']))
    model.eval()

    avg_meter = AverageMeter()
    tongji = [[] for _ in range(21)]
    acc = [AverageMeter() for _ in range(21)]
    rc = [AverageMeter() for _ in range(21)]
    hd = [AverageMeter() for _ in range(21)]
    jacca = AverageMeter()
    sp = [AverageMeter() for _ in range(21)]
    avg_list = [AverageMeter() for _ in range(21)]

    # for c in range(config['num_classes']):
    #     os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, text_idx, organ_features, _ in tqdm(val_loader, total=len(val_loader)):

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

            tp, fp, fn, tn = get_stats(output, target)
            metrics = {
                "accuracy": smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro"),
                "recall": smp.metrics.recall(tp, fp, fn, tn, reduction="micro"),
                "specificity": smp.metrics.specificity(tp, fp, fn, tn, reduction="micro"),
            }
            quality = dict()
            try:
                quality["Hausdorff"] = HD(output, target)
                HDlist.append(quality['Hausdorff'])
                hd[int(organ_features[0])].update(quality["Hausdorff"])
            except:
                HDlist.append(0)
                pass
            dice = dice_coef(output, target)
            Dice.append(dice)
            tongji[organ_features[0]].append(dice)
            iou1 = iou_score(output.unsqueeze(1), target.unsqueeze(1))
            jaccard.append(iou1)
            avg_meter.update(dice, input.size(0))
            acc[int(organ_features[0])].update(metrics['accuracy'])
            rc[int(organ_features[0])].update(metrics['recall'])
            recall.append(float(metrics['recall']))
            sp[int(organ_features[0])].update(metrics['specificity'])

            jacca.update(iou1, input.size(0))
            avg_list[int(organ_features[0])].update(dice)
            break

    ans.HD = HDlist
    ans.Recall = recall
    ans.Jaccard = jaccard
    ans.Dice = Dice
    ans.to_csv('ans.csv', index=False)

    import pickle
    # 保存列表到文件
    with open('organ_list.pkl', 'wb') as file:
        pickle.dump(tongji, file)

    fin_hd = 0
    fin_acc = 0
    fin_rc = 0
    fin_sp = 0
    for i in range(21):
        fin_sp += sp[i].avg
        fin_acc += acc[i].avg
        fin_hd += hd[i].avg
        fin_rc += rc[i].avg
    print('IoU: %.4f' % avg_meter.avg)
    print('jacca: %.4f' % jacca.avg)
    print('hd: %.4f' % (fin_hd/config['num_classes']))
    print('acc: %.4f' % (fin_acc/config['num_classes']))
    print('rc: %.4f' % (fin_rc/config['num_classes']))
    print('sp: %.4f' % (fin_sp/config['num_classes']))
    for j in range(0, 21, 3):
        print(f'IoU{j}: %.4f - IoU{j+1}: %.4f - IoU{j+2}: %.4f'
              % (avg_list[j].avg, avg_list[j+1].avg, avg_list[j+2].avg))


    torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
