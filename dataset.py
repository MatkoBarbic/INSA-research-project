import torch
import torchvision
from PIL import Image
import os
from utils import downsample
import numpy as np

class PanDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_pan, path_to_rgb, path_to_sharp, train_perc, train: bool):
        self.path_to_pan = path_to_pan
        self.path_to_rgb = path_to_rgb
        self.path_to_sharp = path_to_sharp

        self.train = train

        if train:
          self.train_split = round(len(os.listdir(self.path_to_pan)) * train_perc)

          self.pan = sorted(os.listdir(self.path_to_pan))[: self.train_split]
          self.rgb = sorted(os.listdir(self.path_to_rgb))[: self.train_split]
          self.sharp = sorted(os.listdir(self.path_to_sharp))[: self.train_split]

        else:
          self.pan = sorted(os.listdir(self.path_to_pan))[self.train_split: ]
          self.rgb = sorted(os.listdir(self.path_to_rgb))[self.train_split: ]
          self.sharp = sorted(os.listdir(self.path_to_sharp))[self.train_split: ]

        assert len(self.pan) == len(self.rgb) == len(self.sharp)

    def __len__(self):
        if self.train:
          return self.train_split
        else:
          return len(os.listdir(self.path_to_sharp)) - self.train_split

    def __getitem__(self, index):
        # pan_img = torchvision.transforms.functional.to_tensor(Image.open(os.path.join(self.path_to_pan, self.pan[index])))
        # rgb_img = torchvision.transforms.functional.to_tensor(Image.open(os.path.join(self.path_to_rgb, self.pan[index])))
        # sharp_img = torchvision.transforms.functional.to_tensor(Image.open(os.path.join(self.path_to_sharp, self.pan[index])))

        sharp_img = Image.open(os.path.join(self.path_to_sharp, self.sharp[index]))
        pan_img = sharp_img.convert("L")
        rgb_img = downsample(np.array(sharp_img), 2)
        rgb_img = Image.fromarray(rgb_img).resize(sharp_img.size)

        sharp_img = torchvision.transforms.functional.to_tensor(sharp_img)
        pan_img = torchvision.transforms.functional.to_tensor(pan_img)
        rgb_img = torchvision.transforms.functional.to_tensor(rgb_img)

        return (torch.cat([pan_img, rgb_img], dim=0), sharp_img)
