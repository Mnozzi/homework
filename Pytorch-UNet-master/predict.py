import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import logging
from PIL import Image


from evaluate import dice_coeff, mIoU
from utils.data_loading import BasicDataset
from unet import UNet
import pdb

@torch.inference_mode()
def predict(net, dataloader, device, dir_output, mask_values, out_threshold=0.5):
    dice_score = 0
    # TODO 实现对训练好的模型测试，包括数据载入，输入网络进行预测


    # compute the Dice score, ignoring background
    true_masks = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
    # TODO 调用dice_coeff()函数计算DICE值，调用F.cross_entropy()函数计算交叉熵cross-entropy值
    dice_score += dice_coeff()

    # save prediction mask
    mask = mask_pred.argmax(dim=1)
    
    # TODO 调用mask_to_image()，并保存预测mask图像至dir_output, 命名与数据原始名称相同，如：27.tif


    return dice_score

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--dir_img', default='./data/test/img/', help='path of input')
    parser.add_argument('--dir_mask', default='./data/test/mask/', help='path of mask')
    parser.add_argument('--model', '-m', default='./checkpoints/checkpoint_epoch5.pth',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--output', '-o', default='./data/pred/', help='Filenames of output images')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--out_threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    dataset = BasicDataset(Path(args.dir_img), Path(args.dir_mask), args.scale)
    num = len(dataset)
    print(num)
    loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=True, **loader_args)
    dice_score = predict(net=net, dataloader=dataloader, dir_output=args.output, out_threshold=args.out_threshold,
                   device=device, mask_values=mask_values)
    print('Average DICE score on test dataset is:', dice_score/num)






