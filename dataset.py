import torch
import torchvision
from PIL import Image
import os
from utils import downsample
import numpy as np
import cv2

class PanDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_pan, path_to_rgb, path_to_sharp, train_perc, train: bool, downsample_rgb: int = 2):
        self.path_to_pan = path_to_pan
        self.path_to_rgb = path_to_rgb
        self.path_to_sharp = path_to_sharp
        self.downsample_rgb = downsample_rgb

        self.train = train

        if train:
          self.train_split = round(len(os.listdir(self.path_to_sharp)) * train_perc)
          self.sharp = sorted(os.listdir(self.path_to_sharp))[: self.train_split]

        else:
          self.sharp = sorted(os.listdir(self.path_to_sharp))[self.train_split: ]

    def __len__(self):
        if self.train:
          return self.train_split
        else:
          return len(os.listdir(self.path_to_sharp)) - self.train_split

    def __getitem__(self, index):
        sharp_img = cv2.imread(os.path.join(self.path_to_sharp, self.sharp[index]), cv2.IMREAD_UNCHANGED)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)

        rgb_img = downsample(img=sharp_img, factor=self.downsample_rgb)
        pan_img = cv2.cvtColor(sharp_img, cv2.COLOR_RGB2GRAY)
        rgb_img = cv2.resize(rgb_img, (pan_img.shape))

        sharp_img = torchvision.transforms.functional.to_tensor(sharp_img.astype("int16")) / np.amax(sharp_img)
        pan_img = torchvision.transforms.functional.to_tensor(pan_img.astype("int16")) / np.amax(pan_img)
        rgb_img = torchvision.transforms.functional.to_tensor(rgb_img.astype("int16")) / np.amax(rgb_img)

        return (torch.cat([pan_img, rgb_img], dim=0), sharp_img)
