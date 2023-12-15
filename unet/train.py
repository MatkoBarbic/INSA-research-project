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

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs_val = net(x_val)
                loss_val = criterion(outputs_val, y_val)
                val_loss += loss_val.item()
        val_loss /= len(val_loader)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default = '../data/landscapes', help='dataset path')
    parser.add_argument('--batch_size', type=int, default = int(32), help='batch_size')
    parser.add_argument('--lr', type=float, default = float(1e-3), help='learning rate')
    parser.add_argument('--epochs', type=int, default = int(10), help='number of epochs')
    parser.add_argument('--num_workers', type=int, default = int(4), help='number of workers')
 
    args = parser.parse_args()
    data_path = args.data_path
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    num_workers = args.num_workers

    model_folder = "./models/unet/"

    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    
    model_path = os.path.join(model_folder, "unet.pth")
    
    path_to_data = "./data/"
    path_to_train = os.path.join(path_to_data, "train")
    path_to_val = os.path.join(path_to_data, "val")

    path_to_train_pan = os.path.join(path_to_train, "pan")
    path_to_train_rgb = os.path.join(path_to_train, "rgb")
    path_to_train_sharp = os.path.join(path_to_train, "sharp")

    path_to_val_pan = os.path.join(path_to_val, "pan")
    path_to_val_rgb = os.path.join(path_to_val, "rgb")
    path_to_val_sharp = os.path.join(path_to_val, "sharp")

    unet = UNet(n_channels=4, n_classes=3)# .cuda()
    train_data = PanDataset(path_to_pan=path_to_train_pan, path_to_rgb=path_to_train_rgb, path_to_sharp=path_to_train_sharp)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_data = PanDataset(path_to_pan=path_to_val_pan, path_to_rgb=path_to_val_rgb, path_to_sharp=path_to_val_sharp)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    optimizer = optim.Adam(unet.parameters(), lr=lr)
    writer = SummaryWriter('./runs/UNet')
    train_loss, val_loss = train(net=unet, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, model_path=model_path, epochs=epochs, writer=writer)
 
    # x, y = next(iter(train_loader))
 
    # with torch.no_grad():
    #     all_embeddings = []
    #     all_labels = []
    #     for x, y in train_loader:
    #         x , y = x.to(device), y.to(device)
    #         embeddings = unet.get_features(x).view(-1, 128*28*28)
    #         all_embeddings.append(embeddings)
    #         all_labels.append(y)
    #         if len(all_embeddings)>6:
    #             break
    #     embeddings = torch.cat(all_embeddings)
    #     labels = torch.cat(all_labels)
    #     writer.add_embedding(embeddings, label_img=labels, global_step=1)
    #     writer.add_graph(unet, x.to(device))
        
    # Save model weights
    # torch.save(unet.state_dict(), 'unet.pth')