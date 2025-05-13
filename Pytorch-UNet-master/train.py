import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset


dir_img = Path('/kaggle/working/homework/Pytorch-UNet-master/data/train/img')
dir_mask = Path('/kaggle/working/homework/Pytorch-UNet-master/data/train/mask')
dir_checkpoint = Path('/kaggle/working/homework/Pytorch-UNet-master/checkpoints')


def train_model(
        fold,
        model,
        train_loader,
        val_loader,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        early_stop_patience: int = 5
):
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', name=f'fold_{fold}', mode="disabled")
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    n_val = int(len(val_loader) * batch_size)
    n_train = int(len(train_loader) * batch_size)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score

    '''For device==cuda version'''
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    '''For device==cuda version'''

    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_val_score = -np.inf  # 假设监控指标是Dice分数（越大越好）
    epochs_no_improve = 0
    best_model_path = '/kaggle/working/homework/Pytorch-UNet-master/earlymodel'  # 保存最佳模型路径
    # # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                # 数据校验
                assert images.shape[1] == model.n_channels, \
                    f'输入图像的通道数 ({images.shape[1]}) 与模型定义 ({model.n_channels}) 不匹配'

                # 数据搬移到设备
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # 混合精度训练
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)

                # 反向传播
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # 更新进度条和日志
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # --- 验证阶段 (每个epoch结束后执行一次) ---
                val_score, val_ce = evaluate(model, val_loader, device, amp)
                scheduler.step(val_score)  # 仅根据Dice分数调整学习率

                # --- 早停逻辑 ---
                if val_score > best_val_score:
                    best_val_score = val_score
                    epochs_no_improve = 0
                    # 保存最佳模型
                    best_model_path = str(dir_checkpoint / f'best_model_fold{fold}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f'🔥 Epoch {epoch}: 最佳模型已保存, Dice分数={val_score:.4f}')
                else:
                    epochs_no_improve += 1
                    logging.info(f'⏳ Epoch {epoch}: {epochs_no_improve}/{early_stop_patience} 次未提升')

                    # 触发早停
                    if epochs_no_improve >= early_stop_patience:
                        logging.info(f'🛑 Epoch {epoch}: 早停触发!')
                        break  # 终止训练循环

                # --- 常规模型保存 (可选) ---
                if save_checkpoint:
                    checkpoint_path = str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    logging.info(f'📦 Epoch {epoch}: 常规检查点已保存')

                # --- 训练结束后加载最佳模型 ---
            if best_model_path:
                model.load_state_dict(torch.load(best_model_path))
                logging.info(f'🎯 最佳模型已加载: {best_model_path}')
            else:
                logging.warning('⚠️ 未找到最佳模型')

            return model

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=25.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--early-stop', '-es', type=int, default=5,
                        help='Early stopping patience (epochs with no improvement)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    # pdb.set_trace()
    model.to(device=device)

    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, args.scale)

    # # 2. Split into train / validation partitions
    # TODO: 使用K折交叉验证完成模型的训练和验证
    # 定义k值和折数
    k = 5
    num_samples = len(dataset)
    #fold_size = num_samples // k
    # 生成随机索引
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # TODO: 创建每个折对应的Dataset样本索引
    fold_indices = np.array_split(indices, k)

    # 使用每个折进行训练和验证
    for fold in range(k):
        # 每个 Fold 新建模型
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        model.to(device=device)

        if args.load:  # 加载预训练权重
            state_dict = torch.load(args.load, map_location=device)
            del state_dict['mask_values']
            model.load_state_dict(state_dict)

        # 划分数据集
        val_idx = fold_indices[fold].tolist()
        train_idx = np.concatenate([f for i, f in enumerate(fold_indices) if i != fold]).tolist()
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # 创建 DataLoader
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler,
                                num_workers=4, pin_memory=True)

        # 训练模型（优化器在 train_model 内初始化）
        trained_model = train_model(
            fold,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            early_stop_patience=args.early_stop,  # 传递早停参数
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
