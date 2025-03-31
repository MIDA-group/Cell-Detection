"""
 Regression mini-UNet to predict nucleus-likeness, detect local maxima

 Code building on https://github.com/milesial/Pytorch-UNet (GPL-3.0) 
"""

import argparse
import logging
from pathlib import Path
from shutil import copy2

import torch
import torch.nn as nn
import os
import random

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations

#from utils.data_loading import BasicDataset
from utils.utils import *
from unet import UNet

import sys
sys.path.insert(1, '../Utils/')
from data_loader import BasicDataset


IMG_WIDTH = 512
IMG_HEIGHT = 512
BATCH_SIZE = 1
EPOCHS = 450
MAX_TRAIN_IMAGES = 1000

def get_matching_pairs(img_dir, mask_dir, max_images=None):
    """
    Get list of matching image and mask pairs with optional limit
    
    Args:
        img_dir: Directory containing images
        mask_dir: Directory containing masks
        max_images: Maximum number of image pairs to return (if None, return all pairs)
    """
    img_files = set(os.listdir(img_dir))
    mask_files = set(os.listdir(mask_dir))
    
    # Get pairs where corresponding mask exists
    pairs = []
    for img_file in img_files:
        name, ext = os.path.splitext(img_file)
        mask_file = f"{name}_mask{ext}"
        #mask_file = img_file

        if mask_file in mask_files:
            pairs.append(img_file)

    # Sort for reproducibility before sampling
    pairs = sorted(pairs)
    
    # Randomly sample if max_images is specified and less than total available pairs
    if max_images and max_images < len(pairs):
        random.seed(42)  # Set seed for reproducibility
        pairs = random.sample(pairs, max_images)
        pairs = sorted(pairs)
            
    return pairs

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 32,
              learning_rate: float = 1e-6,
              save_checkpoint: bool = True,
              amp: bool = False,
              start_epoch = 1,
              train_img_dir: Path = None,
              train_mask_dir: Path = None,
              val_img_dir: Path = None,
              val_mask_dir: Path = None,
              max_train_images: int = None):
 
    # 1. Create datasets
    transform = albumentations.Compose([
                        albumentations.HorizontalFlip(p=0.5),
                        albumentations.RandomRotate90(p=1),
                        albumentations.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.05),
                        albumentations.GaussianBlur(blur_limit=3,sigma_limit=1),
                        albumentations.GaussNoise(std_range=[0.05,0.1],mean_range=[0,0],per_channel=True,noise_scale_factor=1)
                        ])
    
    # Get matching pairs for training and validation sets
    train_pairs = get_matching_pairs(train_img_dir, train_mask_dir, max_images=MAX_TRAIN_IMAGES)
    if not train_pairs:
        raise RuntimeError(f"No matching image/mask pairs found in {train_img_dir} and {train_mask_dir}")
    
    val_pairs = get_matching_pairs(val_img_dir, val_mask_dir)  
    if not val_pairs:
        raise RuntimeError(f"No matching image/mask pairs found in {val_img_dir} and {val_mask_dir}")
    
    logging.info(f'Found {len(train_pairs)} training and {len(val_pairs)} validation image/mask pairs')
    
    train_dataset = BasicDataset(train_img_dir, train_mask_dir, transform=transform, file_list=train_pairs)
    val_dataset = BasicDataset(val_img_dir, val_mask_dir, transform=None, file_list=val_pairs)

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    grad_scaler = torch.GradScaler(device=device.type,enabled=amp)
    criterion = nn.MSELoss()
    val_criterion = nn.MSELoss(reduction='sum')
    global_step = 0

    train_losses = {}
    val_err = {}
    for epoch in range(start_epoch, epochs+1):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device_type=device.type,enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) 

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': f'{loss.item():.5f}'})
                train_losses[epoch] = epoch_loss / len(train_loader)
                
        if epoch%10==0:
            net.eval()
            err=0.
            count=0
            for batch in val_loader:
                with torch.inference_mode():
                    images = batch['image'].to(device=device, dtype=torch.float32)
                    true_masks = batch['mask'].to(device=device, dtype=torch.float32)
                    with torch.autocast(device_type=device.type,enabled=amp): 
                        masks_pred = net(images)
                        err += val_criterion(masks_pred, true_masks)
                        count += len(batch['image']) 
            val_err[epoch] = err/count
            print(f'Validation error at Epoch {epoch}: {val_err[epoch]:.5f} (MSE over {count} instances)')

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')

    best = min(val_err,key=val_err.get)
    print(f'Best validation score at Epoch {best}: {val_err[best]:.5f}')
    save_training_history(train_losses, val_err)
    save_augmented_samples(train_dataset)
    if save_checkpoint:
        best_model=str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(best))
        copy2(best_model,'MODEL.pth')
        logging.info(f'Copying checkpoint {best_model} to MODEL.pth')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--train-img-dir', type=str, required=True, help='Path to directory containing training images')
    parser.add_argument('--train-mask-dir', type=str, required=True, help='Path to directory containing training masks')
    parser.add_argument('--val-img-dir', type=str, required=True, help='Path to directory containing validation images')
    parser.add_argument('--val-mask-dir', type=str, required=True, help='Path to directory containing validation masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--start-epoch', '-s', type=int, default=1, help='Starting epoch (useful for load)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/', help='Directory to save checkpoints')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    dir_checkpoint = Path(args.checkpoint_dir)

    net = UNet(n_channels=3, n_classes=1)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  amp=args.amp,
                  start_epoch=args.start_epoch,
                  train_img_dir=Path(args.train_img_dir),
                  train_mask_dir=Path(args.train_mask_dir),
                  val_img_dir=Path(args.val_img_dir),
                  val_mask_dir=Path(args.val_mask_dir),
                  max_train_images=MAX_TRAIN_IMAGES)
    except KeyboardInterrupt:
        print('Ctrl-c pressed. Saving model...')
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise