import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from networks.gan import Generator, Discriminator
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument(
    '--lr',
    type = float,
    help ='learning rate',
    default = 0.001
)
parser.add_argument(
    '--batch_size',
    type = int,
    help ='batch size for processing the images',
    default = 32
)
parser.add_argument(
    '--epoch_size',
    type = int,
    help ='no of epochs to train GAN for',
    default = 100
)
args = parser.parse_args()


def train():
    lr = args.lr
    z_dim = 64
    image_dim = 28 * 28 * 1  # 784
    batch_size = args.batch_size
    num_epochs = args.epoch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    discr = Discriminator(image_dim).to(device)
    genr = Generator(z_dim, image_dim).to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
    )   
    dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt_disc = optim.Adam(discr.parameters(), lr=lr)
    opt_gen = optim.Adam(genr.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        with tqdm(loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for batch_idx, (real, _) in enumerate(tepoch):
                real = real.view(-1, 784).to(device)
                batch_size = real.shape[0]
                
                ### Train Discriminator: maximize => log(D(x)) + log(1 - D(G(z)))
                
                ### z = a random noise, x = real sample, D is denoted by the discriminator
                    
                noise = torch.randn(batch_size, z_dim).to(device)
                #print(batch_idx)
                
                fake = genr(noise)
                
                disc_real = discr(real).view(-1)
                
                lossD_real = criterion(disc_real, torch.ones_like(disc_real))
                
                disc_fake = discr(fake).view(-1)
                
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                
                lossD = (lossD_real + lossD_fake) / 2
                
                
                discr.zero_grad()
                lossD.backward(retain_graph=True)
                opt_disc.step()
                
                ## Train Generator
                
                output = discr(fake).view(-1)
                
                lossG = criterion(output, torch.ones_like(output))
                
                genr.zero_grad()
                lossG.backward()
                opt_gen.step()

                tepoch.set_postfix(d_loss=lossD.item(), g_loss=lossG.item())


if __name__ == '__main__':
    train()