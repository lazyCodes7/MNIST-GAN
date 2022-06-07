import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
            
        )
        
    def forward(self, x):
        x = self.disc(x)
        
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, input_size):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, input_size),
            nn.Tanh(),
        
        )
    def forward(self, x):
        x = self.gen(x)
        
        return x