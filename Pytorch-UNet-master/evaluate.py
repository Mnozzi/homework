import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import pdb


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    cross_entropy = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred_logits = net(image)  # 保留原始logits输出
            mask_true_labels = mask_true  # 原始标签不需要one-hot转换

            # 计算交叉熵（使用原始logits和标签）
            cross_entropy += F.cross_entropy(mask_pred_logits, mask_true_labels).item()

            # 转换预测结果为one-hot格式用于Dice计算
            mask_pred = mask_pred_logits.argmax(dim=1)
            mask_pred_oh = F.one_hot(mask_pred, net.n_classes).permute(0, 3, 1, 2).float()
            mask_true_oh = F.one_hot(mask_true_labels, net.n_classes).permute(0, 3, 1, 2).float()

            # 计算Dice系数（排除背景类）
            dice_score += dice_coeff(mask_pred_oh[:, 1:], mask_true_oh[:, 1:]).item()

    net.train()
    return dice_score, cross_entropy


def dice_coeff(input: Tensor, target: Tensor,epsilon=1e-6):
    # Dice loss (objective to minimize) between 0 and 1
    # TODO 完成DICE函数计算DICE值，变量可根据设计需要自行删改
    # 计算交集
    intersection = (input * target).sum(dim=(2, 3))  # 按空间维度求和 → (B, C)

    # 计算并集
    union = input.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    # 计算Dice系数
    dice = (2. * intersection + epsilon) / (union + epsilon)

    # 返回所有样本和类别的平均值
    return dice.mean()

def mIoU(input: Tensor, target: Tensor,epsilon=1e-6):
    # Dice loss (objective to minimize) between 0 and 1
    # TODO 完成mIoU函数计算mIoU值，变量可根据设计需要自行删改
    # 计算交集
    intersection = (input * target).sum(dim=(2, 3))  # (B, C)

    # 计算并集（并集 = A + B - 交集）
    union = input.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

    # 计算IoU
    iou = (intersection + epsilon) / (union + epsilon)

    # 返回所有样本和类别的平均值
    return iou.mean()

