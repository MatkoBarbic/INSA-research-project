import torch
import torchvision
from PIL import Image
import os

class PanDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_pan, path_to_rgb, path_to_sharp):
        self.path_to_pan = path_to_pan
        self.path_to_rgb = path_to_rgb
        self.path_to_sharp = path_to_sharp
        
        self.pan = sorted(os.listdir(self.path_to_pan))
        self.rgb = sorted(os.listdir(self.path_to_rgb))
        self.sharp = sorted(os.listdir(self.path_to_sharp))

        assert len(self.pan) == len(self.rgb) == len(self.sharp)

    def __len__(self):
        return len(os.listdir(self.path_to_sharp))

    def __getitem__(self, index):
        pan_img = torchvision.transforms.functional.to_tensor(Image.open(os.path.join(self.path_to_pan, self.pan[index])))
        rgb_img = torchvision.transforms.functional.to_tensor(Image.open(os.path.join(self.path_to_rgb, self.pan[index])))
        sharp_img = torchvision.transforms.functional.to_tensor(Image.open(os.path.join(self.path_to_sharp, self.pan[index])))

        if pan_img.shape != (1, 410, 410):
            print(self.path_to_pan)
        
        if rgb_img.shape != (3, 410, 410):
            print(self.path_to_rgb)

        return (torch.cat([pan_img, rgb_img], dim=0), sharp_img)