import torch
from torch.utils.data import Dataset
import logging
from utils.transforms import read_data
import numpy as np
# from utils.img_preprocess import add2img_grey


train_idxs = [i for i in range(25)]
test_idxs = [i for i in range(5)]

class CustomDataset(Dataset):
    def __init__(self, is_train=True, data_augment=True):
        self.data_augment = data_augment
        self.is_train = is_train
        # self.target_size = target_size

    def __len__(self):
        if self.is_train:
            return len(train_idxs)
        else:
            return len(test_idxs)

    def __getitem__(self, idx):
        img, label = read_data(idx, is_train=self.is_train, data_augment=self.data_augment)

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        label = torch.from_numpy(label)

        # print(img.shape, label.shape)

        return img, label