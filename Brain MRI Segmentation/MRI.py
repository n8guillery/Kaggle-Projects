import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import random

import glob
import os
import time
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split


DATA_DIR = ".\\data\\lgg-mri-segmentation\\kaggle_3m\\"
NUM_EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



##Data
class MRIDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx]['image'])
        mask = np.float32(cv2.imread(self.df.iloc[idx]['mask'], 0))

        transformed = self.transforms(image=image, mask=mask)

        return transformed['image'], transformed['mask']/255


def import_data(path):
    data = []
    for folder in glob.glob(path + '*'):
        if os.path.isdir(folder):
            folder_name = folder.split('\\')[-1]
            for file in glob.glob(folder + '\\*'):
                if 'mask' in file:
                    image_file = file[:-9] + '.tif'
                    ## Record presence of abnormality
                    mask = np.max(cv2.imread(file))
                    mask = 1 if mask else 0
                    data.append([folder_name, image_file, file, mask])

    df = pd.DataFrame(data, columns=['folder', 'image', 'mask', 'abnormality'])

    return df


def build_loaders(df):
    transforms = A.Compose([
        A.Resize(width=256, height=256, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),

        A.Normalize(),
        ToTensorV2(),
    ])

    train, val = train_test_split(df, stratify=df.abnormality, test_size=0.1)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train, test = train_test_split(train, stratify=train.abnormality, test_size=0.15)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)


    train_dataset = MRIDataset(df=train, transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=26, num_workers=NUM_WORKERS, shuffle=True)

    val_dataset = MRIDataset(df=val, transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=26, num_workers=NUM_WORKERS, shuffle=True)

    test_dataset = MRIDataset(df=test, transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=26, num_workers=NUM_WORKERS, shuffle=True)

    return train_loader, val_loader, test_loader



##Model
class DoubleConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(DoubleConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class Down(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Down, self).__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_chan, out_chan)
        )

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Up, self).__init__()
        self.convtrans = nn.ConvTranspose2d(in_chan, out_chan, 2, stride=2)
        self.conv = DoubleConv(in_chan, out_chan)

    def forward(self, x, x0):
        convtrans = self.convtrans(x)
        cat = torch.cat([x0, convtrans], dim=1)
        result = self.conv(cat)
        return result


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inconv = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024)
        # self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.outconv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.inconv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # x4 = self.down4(x3)
        # x = self.up4(x4, x3)
        x = self.up3(x3, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        x = self.outconv(x)
        x = self.sigmoid(x)
        return torch.squeeze(x)



##Loss
def loss_func(predict, target):
    ## DICE loss
    smooth = 1
    intersection = smooth + 2*((target*predict).sum())
    union = target.sum() + predict.sum()
    diceloss = 1 - intersection/union
    # diceloss = - log(intersection/union)

    ## BCE loss
    bce = nn.BCELoss()
    bceloss = bce(predict, target)

    return bceloss + diceloss



##Training
def train_epoch(model, loss_func, optimizer, loader):
    model.train()
    losses = []

    for data, targets in loader:
        data = data.to(DEVICE)
        predicts = model(data)
        targets = targets.to(DEVICE)
        
        loss = loss_func(predicts, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.array(losses).mean()


def val_step(model, loss_func, loader):
    model.eval()
    losses = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE)
            predicts = model(data)
            targets = targets.to(DEVICE)

            loss = loss_func(predicts, targets)
            losses.append(loss.item())

    return np.array(losses).mean()


def train(model, loss_func, optimizer, train_loader, val_loader, num_epochs=30):
    train_history = []
    val_history = []

    for epoch in range(1,num_epochs+1):
        tic = time.time()
        epoch_loss = train_epoch(model, loss_func, optimizer, train_loader)
        toc = time.time()
        train_history.append(epoch_loss)

        val_loss = val_step(model, loss_func, val_loader)
        val_history.append(val_loss)

        # if epoch % 5 == 0:

        print(f'Epoch {epoch}:')
        print('Train loss =', epoch_loss)
        print('Validation loss =', val_loss)
        print(f'Epoch time = {toc-tic}\n')
    
    return train_history, val_history





def main(plot=True):
    t0 = time.time()
    print('device =', DEVICE)

    df = import_data(DATA_DIR)

    train_loader, val_loader, test_loader = build_loaders(df)

    unet = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

    train_history, val_history = train(unet, loss_func, optimizer, train_loader, val_loader, num_epochs=NUM_EPOCHS)

    test_loss = val_step(unet, loss_func, test_loader)
    tend = time.time()
    totaltime = tend - t0
    print('Test loss =', test_loss)
    print('Total time = ', totaltime)

    # if plot:

    

if __name__ == '__main__':
    main()
