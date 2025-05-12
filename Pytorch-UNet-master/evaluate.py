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

            # predict the mask
            mask_pred = net(image)
            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            # TODO 调用dice_coeff()函数计算DICE值，调用F.cross_entropy()函数计算交叉熵cross-entropy值
            dice_score +=  dice_coeff(mask_pred[:, 1:], mask_true[:, 1:]).item()
            cross_entropy += F.cross_entropy(mask_pred, mask_true).item()

    net.train()
    return dice_score, cross_entropy


def dice_coeff(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    # TODO 完成DICE函数计算DICE值，变量可根据设计需要自行删改


    return

def mIoU(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    # TODO 完成mIoU函数计算mIoU值，变量可根据设计需要自行删改

    return

