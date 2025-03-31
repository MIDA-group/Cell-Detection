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

from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import albumentations

from utils.data_loading import BasicDataset
from utils.utils import *
from unet import UNet



def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-6,
              val_percent: float = 0.1,
              val_frequency: int = 10,
              save_checkpoint: bool = True,
              amp: bool = False,
              start_epoch = 1):

    # 1. Create dataset
    transform = albumentations.Compose([
                        albumentations.HorizontalFlip(p=0.5),
                        albumentations.RandomRotate90(p=1)])

    dataset = BasicDataset(dir_img, dir_mask, transform=transform)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    
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

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.GradScaler(device=device.type,enabled=amp) #Interestingly parameter not named device_type
    criterion = nn.MSELoss()
    val_criterion = nn.MSELoss(reduction='sum')
    global_step = 0

    val_err = {} # Keep track of validation error
    # 5. Begin training
    digits = len(str(epochs))
    for epoch in range(start_epoch, epochs+1):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch:{digits}d}/{epochs}', unit='img') as pbar:
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
                
        if epoch%val_frequency==0:
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

    # 6. Pick best
    best = min(val_err,key=val_err.get)
    print(f'Best validation score at Epoch {best}: {val_err[best]:.5f}')
    if save_checkpoint:
        best_model=str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(best))
        copy2(best_model,args.model)
        logging.info(f'Copyig checkpoint {best_model} to {args.model}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    parser.add_argument('datapath', metavar='DATADIR', help='Directory with "tile" and "mask" subdirectories')
    parser.add_argument('--checkdir', metavar='CHECKDIR', default='./checkpoints', help='Directory of stored network checkpoints')
    parser.add_argument('--checkfreq', '-f', metavar='N', type=int, default=10, help='Store checkpoint every N epochs = validation frequency')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help='Output filename of best network model')
    
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of workers')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')

    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--start-epoch', '-s', type=int, default=1, help='Starting epoch (useful for load)')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()

    # Lazily relying on global vars here; TODO: put into arglist of train_net 
    global dir_img
    dir_img = Path(args.datapath,'tile/')
    global dir_mask
    dir_mask = Path(args.datapath,'mask/')
    global dir_checkpoint
    dir_checkpoint = Path(args.checkdir)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
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
                  val_percent=args.val / 100,
                  val_frequency=args.checkfreq,
                  amp=args.amp,
                  start_epoch=args.start_epoch)
    except KeyboardInterrupt:
        print('Ctrl-c pressed. Saving model...')
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
