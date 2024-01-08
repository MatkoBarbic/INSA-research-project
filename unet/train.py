import torch
import torchvision
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from unet import UNet
from torch.utils.data import DataLoader
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dataset import PanDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(net, val_loader, criterion):
	net.eval()
	val_loss = 0.0
	
	with torch.no_grad():
		for x_val, y_val in val_loader:
			x_val, y_val = x_val.to(device), y_val.to(device)
			outputs_val = net(x_val)
			loss_val = criterion(outputs_val, y_val)
			val_loss += loss_val.item()
	
	val_loss /= len(val_loader)
	
	return val_loss


def train(net, optimizer, train_loader, val_loader, model_path, epochs=5, writer=None):
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = []
        t = tqdm(train_loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {np.mean(train_loss)}')

        val_loss = evaluate(net=unet, val_loader=val_loader, criterion=criterion)
        print(f'validation loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), model_path)

        if writer is not None:
            writer.add_scalar('training loss', np.mean(train_loss), epoch)
            writer.add_scalar('validation loss', val_loss, epoch)

            # img_grid = torchvision.utils.make_grid(outputs[:16].detach().cpu())
            # writer.add_image('colorized', img_grid, epoch)
            # img_grid = torchvision.utils.make_grid(y[:16].detach().cpu())
            # writer.add_image('original', img_grid, epoch)

    return np.mean(train_loss), val_loss


if __name__=='__main__':
	torch.cuda.empty_cache()
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, default = '../data/landscapes', help='dataset path')
	parser.add_argument('--batch_size', type=int, default = int(4), help='batch_size')
	parser.add_argument('--lr', type=float, default = float(1e-3), help='learning rate')
	parser.add_argument('--epochs', type=int, default = int(10), help='number of epochs')
	parser.add_argument('--num_workers', type=int, default = int(4), help='number of workers')

	args = parser.parse_args()
	data_path = args.data_path
	batch_size = args.batch_size
	lr = args.lr
	epochs = args.epochs
	num_workers = args.num_workers
	
	train_perc = 0.8

	model_folder = "./models/unet/"

	if not os.path.isdir(model_folder):
		os.makedirs(model_folder)

	model_path = os.path.join(model_folder, "unet.pth")

	path_to_data = "./data/"
	# path_to_data = path_to_data + "SOTA/"
	path_to_data = path_to_data + "Tiled Pleiades images/sharp/"
      

	# if "SOTA" not in path_to_data:
	# 	path_to_train = os.path.join(path_to_data, "train")
	# 	path_to_val = os.path.join(path_to_data, "val")

	# 	path_to_train_pan = os.path.join(path_to_train, "pan")
	# 	path_to_train_rgb = os.path.join(path_to_train, "rgb")
	# 	path_to_train_sharp = os.path.join(path_to_train, "sharp")

	# 	path_to_val_pan = os.path.join(path_to_val, "pan")
	# 	path_to_val_rgb = os.path.join(path_to_val, "rgb")
	# 	path_to_val_sharp = os.path.join(path_to_val, "sharp")

	# else:
	path_to_train_pan = os.path.join(path_to_data, "pan")
	path_to_train_rgb = os.path.join(path_to_data, "rgb")
	path_to_train_sharp = os.path.join(path_to_data, "sharp")

	path_to_val_pan = os.path.join(path_to_data, "pan")
	path_to_val_rgb = os.path.join(path_to_data, "rgb")
	path_to_val_sharp = os.path.join(path_to_data, "sharp")

	unet = UNet(n_channels=4, n_classes=3).cuda()
	# train_data = PanDataset(path_to_pan=path_to_train_pan, path_to_rgb=path_to_train_rgb, path_to_sharp=path_to_train_sharp)
	# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	# val_data = PanDataset(path_to_pan=path_to_val_pan, path_to_rgb=path_to_val_rgb, path_to_sharp=path_to_val_sharp)
	# val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	train_data = PanDataset(path_to_pan=path_to_train_pan, path_to_rgb=path_to_train_rgb, path_to_sharp=path_to_train_sharp, train_perc=train_perc, train=True)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	val_data = PanDataset(path_to_pan=path_to_val_pan, path_to_rgb=path_to_val_rgb, path_to_sharp=path_to_val_sharp, train_perc=1 - train_perc, train=True)
	val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	optimizer = optim.Adam(unet.parameters(), lr=lr)
	writer = SummaryWriter('./runs/UNet')
	print(torch.cuda.device_count())
	train_loss, val_loss = train(net=unet, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, model_path=model_path, epochs=epochs, writer=writer)
