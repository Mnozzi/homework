import argparse
import logging
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset

# 数据路径定义
dir_img = Path('/kaggle/working/homework/Pytorch-UNet-master/data/train/img')
dir_mask = Path('/kaggle/working/homework/Pytorch-UNet-master/data/train/mask')
dir_checkpoint = Path('/kaggle/working/homework/Pytorch-UNet-master/checkpoints')


def train_model(
        fold: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
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
) -> nn.Module:
    """
    K折交叉验证的单Fold训练函数
    """
    # 初始化WandB（已禁用）
    experiment = wandb.init(project='U-Net', name=f'fold_{fold}', mode="disabled")
    experiment.config.update({
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'val_percent': val_percent,
        'img_scale': img_scale,
        'amp': amp
    })

    # 优化器和学习率调度器
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    criterion = nn.CrossEntropyLoss()

    # 早停变量
    best_val_score = -np.inf
    epochs_no_improve = 0
    best_model_path = None

    # 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        global_step = 0

        # --- 训练阶段 ---
        with tqdm(total=len(train_loader), desc=f'Fold {fold} Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                # 数据校验和设备搬移
                assert images.shape[1] == model.n_channels, \
                    f"输入通道数 {images.shape[1]} 与模型定义 {model.n_channels} 不匹配"
                images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device, dtype=torch.long)

                # 混合精度训练
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)

                # 反向传播
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                # 更新进度条
                pbar.update(1)
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})
                experiment.log({'train_loss': loss.item(), 'step': global_step, 'epoch': epoch})

        # --- 验证阶段 (每个Epoch结束后执行一次) ---
        val_score, val_ce = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)  # 学习率调整
        logging.info(f'Fold {fold} Epoch {epoch}: Val Dice = {val_score:.4f}, CE Loss = {val_ce:.4f}')

        # --- 早停逻辑 ---
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            best_model_path = str(dir_checkpoint / f'best_model_fold{fold}.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Fold {fold}: 保存最佳模型至 {best_model_path}')
        else:
            epochs_no_improve += 1
            logging.info(f'Fold {fold}: {epochs_no_improve}/{early_stop_patience} 次未提升')
            if epochs_no_improve >= early_stop_patience:
                logging.info(f'Fold {fold}: 早停触发!')
                break  # 终止当前Fold的训练

        # --- 保存常规检查点 ---
        if save_checkpoint:
            checkpoint_path = str(dir_checkpoint / f'checkpoint_fold{fold}_epoch{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)

    # 加载最佳模型
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logging.info(f'Fold {fold}: 已加载最佳模型')
    else:
        logging.warning(f'Fold {fold}: 未找到最佳模型')

    return model


def get_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练UNet模型')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='训练轮次')
    parser.add_argument('--batch-size', '-b', type=int, default=2, help='批量大小')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help='学习率')
    parser.add_argument('--load', '-f', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='图像缩放比例')
    parser.add_argument('--validation', '-v', type=float, default=25.0, help='验证集比例（0-100）')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')
    parser.add_argument('--bilinear', action='store_true', help='使用双线性上采样')
    parser.add_argument('--classes', '-c', type=int, default=2, help='分类类别数')
    parser.add_argument('--early-stop', '-es', type=int, default=5, help='早停耐心值')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'训练设备: {device}')

    # 初始化K折交叉验证
    k = 5
    dataset = BasicDataset(dir_img, dir_mask, args.scale)
    indices = np.random.permutation(len(dataset))
    fold_indices = np.array_split(indices, k)

    # 创建检查点目录
    dir_checkpoint.mkdir(parents=True, exist_ok=True)

    # K折训练循环
    for fold in range(k):
        logging.info(f'\n{"=" * 40}')
        logging.info(f'开始训练 Fold {fold + 1}/{k}')
        logging.info(f'{"=" * 40}\n')

        # 数据划分
        val_idx = fold_indices[fold]
        train_idx = np.concatenate([fold_indices[i] for i in range(k) if i != fold])

        # 创建数据加载器
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(val_idx),
            num_workers=4,
            pin_memory=True
        )

        # 初始化模型
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear).to(device)
        if args.load:
            state_dict = torch.load(args.load, map_location=device)
            if 'mask_values' in state_dict:
                del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'从 {args.load} 加载预训练权重')

        # 训练当前Fold
        trained_model = train_model(
            fold=fold,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stop_patience=args.early_stop,
            amp=args.amp
        )

        # 保存最终模型（可选）
        final_model_path = str(dir_checkpoint / f'final_model_fold{fold}.pth')
        torch.save(trained_model.state_dict(), final_model_path)