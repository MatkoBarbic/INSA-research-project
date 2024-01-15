import os, time, pickle, argparse, network, util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dataset import PanDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(G, D, val_loader, criterion, L1_lambda):
    G.eval()
    D.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)

            if x_val.size()[2] != opt.input_size:
                x_val = util.imgs_resize(x_val.cpu(), opt.input_size).to(device)
                y_val = util.imgs_resize(y_val.cpu(), opt.input_size).to(device)
            
            # Evaluate the generator
            G_result = G(x_val)
            D_result = D(x_val, G_result).squeeze()

            # print(G_result.size())
            # print(D_result.size())
            # print(y_val)

            D_val_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))
            G_val_loss = criterion(G_result, y_val)
            
            G_combined_loss = D_val_loss + opt.L1_lambda * G_val_loss

            val_loss += G_combined_loss.item()

    val_loss /= len(val_loader)

    return val_loss, G_val_loss.item(), D_val_loss.item()



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades',  help='')
parser.add_argument('--train_subfolder', required=False, default='train',  help='')
parser.add_argument('--test_subfolder', required=False, default='val',  help='')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--criterion', type=str, default="MSE", help='which loss function to use for the generator. Options are "MSE" or "L1')

parser.add_argument('--input_size', type=int, default=512, help='input size')
parser.add_argument('--crop_size', type=int, default=512, help='crop size (0 is false)')
parser.add_argument('--resize_scale', type=int, default=560, help='resize scale (0 is false)')

parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')
parser.add_argument('--train_epoch', type=int, default=200, help='number of train epochs')

parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--L1_lambda', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
opt = parser.parse_args()
print(opt)

## Results save path
path_to_data = "../../data/"
# path_to_data = path_to_data + "SOTA/"
path_to_data = path_to_data + "Tiled Pleiades images/"

model_folder = "./models/pix2pix/"
train_hist_folder = "./train_hist/pix2pix/"

if "SOTA" in path_to_data:
    train_hist_folder = os.path.join(train_hist_folder, "SOTA")
    model_folder = os.path.join(model_folder, "SOTA")
    g_model_path = os.path.join(model_folder, "g_pix2pix_SOTA.pth")
    d_model_path = os.path.join(model_folder, "d_pix2pix_SOTA.pth")

    # opt.input_size = 512
    # opt.input_size = 512
    # opt.input_size = 560

else:
    train_hist_folder = os.path.join(train_hist_folder, "Pleiades")
    model_folder = os.path.join(model_folder, "")
    g_model_path = os.path.join(model_folder, "g_pix2pix_pleiades.pth")
    d_model_path = os.path.join(model_folder, "d_pix2pix_pleiades.pth")

## Check gpus
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    print(f"Number of available GPUs: {num_gpus}")
else:
    print("No GPUs available on this system.")

# data_loader
train_perc = 0.8

if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

if "SOTA" in path_to_data:
    downscale_factor = 5
else:
    downscale_factor = 2
    
path_to_train_pan = os.path.join(path_to_data, "pan")
path_to_train_rgb = os.path.join(path_to_data, "rgb")
path_to_train_sharp = os.path.join(path_to_data, "sharp")

path_to_val_pan = os.path.join(path_to_data, "pan")
path_to_val_rgb = os.path.join(path_to_data, "rgb")
path_to_val_sharp = os.path.join(path_to_data, "sharp")

train_data = PanDataset(path_to_pan=path_to_train_pan, path_to_rgb=path_to_train_rgb, path_to_sharp=path_to_train_sharp, train_perc=train_perc, train=True, downsample_rgb=downscale_factor)
train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

test_data = PanDataset(path_to_pan=path_to_val_pan, path_to_rgb=path_to_val_rgb, path_to_sharp=path_to_val_sharp, train_perc=1 - train_perc, train=True, downsample_rgb=downscale_factor)
test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

test = test_loader.__iter__().__next__()[0]
img_size = test.size()[2]

# network
G = network.generator(opt.ngf)
D = network.discriminator(opt.ndf)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
G.train()
D.train()

# loss
BCE_loss = nn.BCELoss().cuda()

if opt.criterion == "MSE":
    criterion = torch.nn.MSELoss()
else:
    L1_loss = nn.L1Loss().cuda()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()

best_test_loss = float('inf')

for epoch in range(opt.train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    num_iter = 0
    for x_, y_ in tqdm(train_loader):
        # train discriminator D
        D.zero_grad()
            
        if img_size != opt.input_size:
            x_ = util.imgs_resize(x_, opt.input_size)
            y_ = util.imgs_resize(y_, opt.input_size)

        if opt.resize_scale:
            x_ = util.imgs_resize(x_, opt.resize_scale)
            y_ = util.imgs_resize(y_, opt.resize_scale)

        if opt.crop_size:
            x_, y_ = util.random_crop(x_, y_, opt.crop_size)

        if opt.fliplr:
            x_, y_ = util.random_fliplr(x_, y_)

        x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())

        D_result = D(x_, y_).squeeze()
        D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))

        G_result = G(x_)
        D_result = D(x_, G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda()))

        D_train_loss = (D_real_loss + D_fake_loss) * 0.5
        D_train_loss.backward()
        D_optimizer.step()

        train_hist['D_losses'].append(D_train_loss.item())

        D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        G_result = G(x_)
        D_result = D(x_, G_result).squeeze()

        G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) + opt.L1_lambda * criterion(G_result, y_)
        G_train_loss.backward()
        G_optimizer.step()

        train_hist['G_losses'].append(G_train_loss.item())

        G_losses.append(G_train_loss.item())

        val_loss, G_test_loss, D_test_loss = evaluate(G=G, D=D, val_loader=test_loader, criterion=criterion, L1_lambda=opt.L1_lambda)

        if val_loss < best_test_loss:
            torch.save(G.state_dict(), g_model_path)
            torch.save(D.state_dict(), d_model_path)

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, train_loss_d: %.7f, train_loss_g: %.7f' % ((epoch + 1), opt.train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    
    print('[%d/%d] - val_loss_d: %.7f, val_loss_g: %.7f' % ((epoch + 1), opt.train_epoch, torch.FloatTensor(D_test_loss),
                                                              torch.FloatTensor(G_test_loss)))
    
    # fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    # util.show_result(G, Variable(fixed_x_.cuda(), volatile=True), fixed_y_, (epoch+1), save=True, path=fixed_p)
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.train_epoch, total_ptime))
print("Training finish!... save training results")

# torch.save(G.state_dict(), g_model_path)
# torch.save(D.state_dict(), d_model_path)

# torch.save(G.state_dict(), root + model + 'generator_param.pkl')
# torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')

with open(os.path.join(train_hist_folder, 'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)

# util.show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
# util.generate_animation(root, model, opt)
