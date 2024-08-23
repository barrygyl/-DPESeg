# 人工智能
# 项目：
# 开发人：Barry
# 开发时间：2023-05-31  20:32
# 开发工具：PyCharm

import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import hd95 as hd

'''此部分为模型的评估方式'''


# iou计算，交并比
def iou_score(output, target, threshold=0.5):
    smooth = 1e-5

    assert output.shape == target.shape, "Output and target must have the same shape."
    assert output.dtype == torch.float32 and target.dtype == torch.float32, "Output and target must be float32 tensors."
    assert len(output.shape) == 4 and output.shape[1] == 1, "Output and target must have shape (batch_size, 1, " \
                                                            "height, width). "

    device = output.device

    output_ = (output > threshold).to(torch.float32).to(device)
    target_ = (target > threshold).to(torch.float32).to(device)
    intersection = torch.sum(output_ * target_)
    union = torch.sum(output_ + target_ > 0)

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


# # Dice系数
def dice_coef(output, target):
    smooth = 1e-5

    assert output.shape == target.shape, "Output and target must have the same shape."
    assert output.dtype == torch.float32 and target.dtype == torch.float32, "Output and target must be float32 tensors."

    device = output.device

    output = torch.sigmoid(output).to(device)
    # print(output)
    target = target.to(device)

    intersection = torch.sum(output * target, dim=(1, 2))
    # union = torch.sum(output + target, dim=(1, 2))
    union = torch.sum(output, dim=(1, 2)) + torch.sum(target, dim=(1, 2))

    dice = (2. * intersection + smooth) / (union + smooth)

    return dice.mean().item()


'''后续可继续加新评估标准'''
# def dice_coef(output, target):
#     smooth = 1e-5

#     assert output.shape == target.shape, "Output and target must have the same shape."
#     assert output.dtype == torch.float32 and target.dtype == torch.float32, "Output and target must be float32 tensors."

#     device = output.device

#     output = torch.sigmoid(output).to(device)
#     # print(output)
#     target = target.to(device)

#     intersection = torch.sum(output * target)
#     # union = torch.sum(output + target, dim=(1, 2))
#     union = torch.sum(output) + torch.sum(target)

#     dice = (2. * intersection + smooth) / (union + smooth)

#     return dice.item()



def get_stats(output, target):
    output = torch.sigmoid(output)
    output_binary = (output > 0.5).float()

    tp = torch.sum((output_binary == 1) & (target == 1))
    fp = torch.sum((output_binary == 1) & (target == 0))
    fn = torch.sum((output_binary == 0) & (target == 1))
    tn = torch.sum((output_binary == 0) & (target == 0))

    return tp, fp, fn, tn


def HD(output, target):
    output = torch.sigmoid(output)
    output_binary = (output > 0.5)

    output_binary = output_binary.squeeze().cpu().numpy().astype(np.uint8)
    target = target.squeeze().cpu().numpy().astype(np.uint8)

    ans = hd(output_binary, target)
    # predicted_indices = np.argwhere(output_binary > 0)
    # target_indices = np.argwhere(target > 0)
    #
    # hd = max(directed_hausdorff(predicted_indices, target_indices)[0],
    #          directed_hausdorff(target_indices, predicted_indices)[0])

    return ans