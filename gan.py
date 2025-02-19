# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 00:41:14 2024

@author: 37092
"""
'''
import argparse
import os
import numpy as np
import math
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=304, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

val_percent: float = 0.1
batch_size: int = 200
n_epochs: int = 1#40
lr: float = 0.0002


img_shape = (opt.channels, 1, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class PrepareDataset(Dataset):
    def __init__(self, data_file: str, mask_file: str, ):
        self.data_file = Path(data_file)
        self.mask_file = Path(mask_file)

        df_data = pd.read_csv(self.data_file)
        df_mask = pd.read_csv(self.mask_file)
        
        self.data_tensor = torch.tensor(df_data.values).unsqueeze(1).float()
        self.mask_tensor = torch.tensor(df_mask.values).unsqueeze(1).float()
    
    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, idx):
        data = self.data_tensor[idx]
        mask = self.mask_tensor[idx]

        return {
            'data': data.contiguous(),
            'mask': mask.contiguous()
        }


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(304, 128, normalize=False),
            *block(128, 256),
            nn.Linear(1024, 2*int(np.prod(img_shape))),
            # nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        time_series = img[:, :int(np.prod(img_shape))]
        label = img[:, int(np.prod(img_shape)):]
        # img = img.view(img.size(0), *img_shape)
        return time_series, label


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape))*2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, time_series, label):
        x = torch.cat((time_series, label), dim=1)
        # img_flat = img.view(img.size(0), -1)
        validity = self.model(x)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# 1. Create dataset
dataset = PrepareDataset('./data_add_noise/zx_normalized.csv', './data_add_noise/zy.csv')

# 2. Split into train / validation partitions
n_val = int(len(dataset) * val_percent)#
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# 3. Create data loaders
loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, msks) in enumerate(train_loader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0)+msks.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0)+msks.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        real_msks = Variable(msks.type(Tensor))
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs, gen_segs = generator(real_imgs)#z

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_segs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # real_loss = adversarial_loss(discriminator(real_imgs), valid)
        real_loss = adversarial_loss(discriminator(real_imgs, real_msks), valid)
        
        # fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_msks.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        
'''
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=304, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

val_percent: float = 0.1
batch_size: int = opt.batch_size
n_epochs: int = opt.n_epochs
lr: float = opt.lr

img_shape = (opt.channels, 1, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class PrepareDataset(Dataset):
    def __init__(self, data_file: str, mask_file: str):
        self.data = pd.read_csv(data_file).values
        self.labels = pd.read_csv(mask_file).values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        time_series = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(time_series, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(304, 128, normalize=False),
            *block(128, 256),
            nn.Linear(256, 2 * int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        time_series = img[:, :int(np.prod(img_shape))]
        label = img[:, int(np.prod(img_shape)):]
        return time_series, label

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, time_series, label):
        x = torch.cat((time_series, label), dim=1)
        validity = self.model(x)
        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# 1. Create dataset
dataset = PrepareDataset('./data_add_noise/zx_normalized.csv', './data_add_noise/zy.csv')

# # 2. Split into train / validation partitions
# n_val = int(len(dataset) * val_percent)
# n_train = len(dataset) - n_val
# train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# # 3. Create data loaders
# loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
# train_loader = DataLoader(train_set, shuffle=True, **loader_args)
# val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

# Initialize lists to store losses
G_losses = []
D_losses = []

for epoch in range(opt.n_epochs):
    for i, (real_time_series, real_labels) in enumerate(dataloader):
        # real_imgs = batch['data'].to(device)
        # real_msks = batch['mask'].to(device)
        real_imgs = real_time_series.to(device)
        real_msks = real_labels.to(device)

        # Adversarial ground truths
        valid = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(real_imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images and labels
        gen_imgs, gen_msks = generator(real_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_msks), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs, real_msks), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_msks.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


        # Store the losses
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i

    #     # Save generated samples periodically
    #     if i % opt.sample_interval == 0:
    #         plt.cla()
    #         plt.plot(gen_imgs.detach().numpy()[0], c='red', lw=3, label='Generated data')
    #         plt.plot(real_imgs.detach().numpy()[0], c='black', lw=1, label='Real data')
    #         # plt.text(1, .5, 'the prob of generated data is real = %.2f' % prob_fake.data.numpy().mean())
    #         # plt.ylim((-1.1, 1.1))
    #         # plt.legend(loc='best', fontsize=10)
    #         plt.draw()
    #         plt.pause(0.01) 
    #         # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
    # plt.ioff()
    # plt.show()      
    
    # print(prob_real.mean())
    # print(prob_fake.mean())
    # print('-----------------------------------------------')
    # if (torch.abs(prob_real.mean() - 0.5) <= 1.e-5):
    #     break    
# Plot the losses

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()        