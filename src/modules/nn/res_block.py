import torch.nn as nn
import torch.nn.functional as F

from .san_block import conv1x1

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def deconv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        channel = channel_out // 4

        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel_out, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channel_out)

        self.shortcut = self._shortcut(channel_in, channel_out)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return F.relu(x + self.shortcut(x)) # skip connection

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=1)


class ResNet(nn.Module):
    def __init__(self, block_num):
        super().__init__()
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        c = 32
        net = [
            conv1x1(1, c),
            nn.BatchNorm2d(c),
            relu,
        ]

        for i in range(block_num):
            net.append(maxpool)
            if i == 0:
                net.append(conv1x1(c, c))
            else:
                net.append(conv1x1(c // 2, c))
            net.append(ResBlock(c, c))
            net.append(nn.BatchNorm2d(c))
            net.append(relu)
            c *= 2

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
