import torch
import torchvision
from PIL import Image
import os

class PanDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_pan, path_to_rgb, path_to_sharp):
        self.path_to_pan = path_to_pan
        self.path_to_rgb = path_to_rgb
        self.path_to_sharp = path_to_sharp

        # TODO: Make sure the pan, rgb and sharp images correspond to the same image
        
        self.pan = os.listdir(self.path_to_pan)
        self.rgb = os.listdir(self.path_to_rgb)
        self.sharp = os.listdir(self.path_to_sharp)

        assert len(self.pan) == len(self.rgb) == len(self.sharp)

    def __len__(self):
        return len(os.listdir(self.path_to_sharp))

    def __getitem__(self, index):
        pan_img = torchvision.transforms.functional.to_tensor(os.path.join(self.path_to_pan, Image.open(self.pan[index])))
        rgb_img = torchvision.transforms.functional.to_tensor(os.path.join(self.path_to_rgb, Image.open(self.rgb[index])))
        sharp_img = torchvision.transforms.functional.to_tensor(os.path.join(self.path_to_sharp, Image.open(self.sharp[index])))

        return (torch.cat([pan_img, rgb_img], dim=1), sharp_img)