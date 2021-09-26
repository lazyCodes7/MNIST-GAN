import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class DCGenerator(nn.Module):
    def __init__(self, rand_noise, out_channels, feature_size):
        super(DCGenerator, self).__init__()
        self.gnet = nn.Sequential(
            self._block(rand_noise, feature_size*16, kernel_size = 4, stride = 1, padding = 0),
            self._block(feature_size*16, feature_size*8, kernel_size = 4, stride = 2, padding = 1),
            self._block(feature_size*8, feature_size*4, kernel_size = 4, stride = 2, padding = 1),
            self._block(feature_size*4, feature_size*2, kernel_size = 4, stride = 2, padding = 1),
        )
        
        self.out = nn.Sequential(
            nn.ConvTranspose2d(feature_size*2, out_channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.ReLU()
        
        )
    def forward(self, x):
        x = self.gnet(x)
        x = self.out(x)
        
        return x

class DCDiscriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(DCDiscriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)
        
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)