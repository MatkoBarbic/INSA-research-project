import torch
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(generator, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(4, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)

        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 2)
        self.deconv2 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d)

        self.deconv3 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))

        d1 = self.deconv1_bn(self.deconv1(F.relu(e3)))
        d1 = torch.cat([d1, e2], 1)
        d2 = self.deconv2_bn(self.deconv2(F.relu(d1)))
        d2 = torch.cat([d2, e1], 1)
        d3 = self.deconv3(F.relu(d2))

        o = F.tanh(d3)

        return o


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(7, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        # self.conv3_bn = nn.BatchNorm2d(d * 4)
        # self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        # self.conv4_bn = nn.BatchNorm2d(d * 8)
        # self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        # x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        # x = F.sigmoid(self.conv5(x))
        x = F.sigmoid(self.conv3(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()