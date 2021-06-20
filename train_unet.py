import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from unet import UNet, UNet_2div
from utils.metrics import *
from PIL import Image
from std_criterion import V_std

from torch.utils.tensorboard import SummaryWriter
# from utils.dataset import BasicDataset
from utils.custom_dataset import CustomDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    trainset = CustomDataset(data_augment=True)
    testset = CustomDataset(is_train=False, data_augment=False)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250], gamma=0.1)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
        # criterion = lambda x, y: weighted_BCE_loss(x, y)
    else:
        # criterion = nn.BCEWithLogitsLoss()
         criterion = lambda x, y: weighted_BCE_loss(x, y)
    
    best_iou = 0
    best_vinfo = 0
    best_vrand = 0

    losses = []
    ious = []
    val_ious = []
    vrands_val = []
    vinfo_val = []
    for epoch in range(epochs):
        net.train()
        IoU = AverageMeter()
        Acc = AverageMeter()
        Loss = AverageMeter()

        epoch_loss = 0
        with tqdm(total=25, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, labels in train_loader:
                
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = labels.to(device=device)

                masks_pred = net(imgs).squeeze(1)
                masks_pred = torch.sigmoid(masks_pred)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.2)
                optimizer.step()
                
                masks_pred = masks_pred.detach().cpu().numpy()
                true_masks = true_masks.cpu().numpy()
                iou, size = calc_IoU(masks_pred, true_masks)
                IoU.update(iou, size)
                acc, size = calc_Acc(masks_pred, true_masks)
                Acc.update(acc, size)
                Loss.update(loss, size)

                pbar.update(imgs.shape[0])
                global_step += 1

        lr_scheduler.step()
        print('IoU in train set', IoU.avg, 'Acc in train set', Acc.avg)
        tmp_iou, tmp_vinfo, tmp_vrand = valid(net, device, val_loader)

        losses.append(Loss.avg)
        ious.append(IoU.avg)
        val_ious.append(tmp_iou)
        vrands_val.append(tmp_vrand)
        vinfo_val.append(tmp_vinfo)

        if best_iou < tmp_iou:
            best_iou = tmp_iou
        
        if best_vinfo < tmp_vinfo:
            best_vinfo = tmp_vinfo
        
        if best_vrand < tmp_vrand:
            best_vrand = tmp_vrand

        if (epoch+1)%50==0 and save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    print(f'Best result in Test set: Iou: {best_iou}, V_info: {best_vinfo}, V_rand: {best_vrand}')
    with open('train_res.txt', 'w') as f:
        write_lists = \
            {'Loss': losses, 'Train iou': ious, 'Val iou': val_ious, 'val vrands': vrands_val, 'val_info': vinfo_val}
        for key, items in write_lists.items():
            f.write(f'{key} ')
            for item in items:
                f.write(f'{item},')
    writer.close()


def valid(net, device, loader):
    """Evaluation without the densecrf with the dice coefficient"""
    target_size = (256, 256)
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch

    IoU = AverageMeter()
    Acc = AverageMeter()
    VInfo = AverageMeter()
    VRand = AverageMeter()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = labels.to(device=device)

            with torch.no_grad():
                mask_pred = net(imgs)
                true_masks = true_masks.unsqueeze(1)
                true_masks = F.interpolate(true_masks, size=target_size, mode='nearest')
                mask_pred = F.interpolate(mask_pred, size=target_size, mode='bilinear', align_corners=True)
                mask_pred = torch.sigmoid(mask_pred)
            
            draw_label = mask_pred[0, 0].detach().cpu().numpy()
            draw_label = Image.fromarray((draw_label * 255).astype(np.uint8))
            draw_label.save(f'pred_{i}.png')

            mask_pred = mask_pred.detach().cpu().numpy()
            true_masks = true_masks.cpu().numpy()

            iou, s = calc_IoU(mask_pred, true_masks)
            IoU.update(iou, s)
            acc, s = calc_Acc(mask_pred, true_masks)
            Acc.update(acc, s)
            vinfo, vrand = V_std(mask_pred[0, 0], true_masks[0, 0])
            VInfo.update(vinfo, 1)
            VRand.update(vrand, 1)

            pbar.update()

    print('IoU in valid set', IoU.avg, 'V_info in test set', VInfo.avg, 'V_rand', VRand.avg)
    return IoU.avg, VInfo.avg, VRand.avg


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0003,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    # net = UNet(n_channels=3, n_classes=1, bilinear=True)
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')
    # path = 'pretrained/unet_carvana_scale1_epoch5.pth'
    # net.load_state_dict(torch.load(path))
    
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
