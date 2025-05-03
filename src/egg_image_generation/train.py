"""Trains a conditional generative adversarial network on a dataset.
Based on https://github.com/qbxlvnf11/conditional-GAN.
"""

import logging
import os
import re
from typing import List, Tuple

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torch import Tensor, autograd
from torch.autograd import Variable
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from train_get_args import get_args

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%I:%M:%S %p')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

#   Based on    https://github.com/qbxlvnf11/conditional-GAN/blob/main/conditional-GAN-generating-fashion-mnist.ipynb
class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_size, img_shape, class_num):
        super().__init__()
        
        self.z_size = z_size
        self.img_shape = img_shape
        self.output_dim = img_shape[0] * img_shape[1] * img_shape[2]

        self.label_emb = nn.Embedding(class_num, class_num)
        
        self.model = nn.Sequential(
            nn.Linear(self.z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], self.output_dim),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        
        # Reshape z
        z = z.view(-1, self.z_size)
        
        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        
        # Concat image & label
        x = torch.cat([z, c], 1)
        
        # Generator out
        out = self.model(x)
        
        # [Batch Size, Channels, Width, Height]
        return out.view(-1, *self.img_shape)    

#   Based on    https://github.com/qbxlvnf11/conditional-GAN/blob/main/conditional-GAN-generating-fashion-mnist.ipynb
class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, img_shape, class_num):
        super().__init__()
        
        self.label_emb = nn.Embedding(class_num, class_num)
        self.input_dim = img_shape[0] * img_shape[1] * img_shape[2]
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim + class_num, discriminator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[2], 1),
            # nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        
        # Reshape fake image
        x = x.view(-1, self.input_dim)
        
        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        
        # Concat image & label
        x = torch.cat([x, c], 1)
        
        # Discriminator out
        out = self.model(x)
        
        return out.squeeze()

class ConvGenerator(nn.Module):
    def __init__(self, generator_layer_size, z_size, img_shape, class_num):
        super().__init__()
        self.z_size = z_size
        self.img_shape = img_shape
        self.channels, self.height, self.width = img_shape
        self.class_num = class_num
        self.label_emb = nn.Embedding(class_num, class_num)

        self.init_size = self.height // 4
        self.project = nn.Linear(z_size + class_num, generator_layer_size[0] * self.init_size * self.init_size)

        self.model = nn.Sequential(
            nn.BatchNorm2d(generator_layer_size[0]),
            nn.ConvTranspose2d(generator_layer_size[0], generator_layer_size[1], 4, 2, 1),
            nn.BatchNorm2d(generator_layer_size[1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(generator_layer_size[1], self.channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.project(x).view(-1, self.model[0].num_features, self.init_size, self.init_size)
        return self.model(x)

class ConvDiscriminator(nn.Module):
    def __init__(self, discriminator_layer_size, img_shape, class_num):
        super().__init__()
        self.img_shape = img_shape
        self.channels, self.height, self.width = img_shape
        self.class_num = class_num
        self.label_emb = nn.Embedding(class_num, self.channels * self.height * self.width)

        self.model = nn.Sequential(
            nn.Conv2d(self.channels * 2, discriminator_layer_size[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_layer_size[0], discriminator_layer_size[1], 4, 2, 1),
            nn.BatchNorm2d(discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear((self.height // 4) * (self.width // 4) * discriminator_layer_size[1], 1)
        )

    def forward(self, x, labels):
        c = self.label_emb(labels).view(-1, self.channels, self.height, self.width)
        x = torch.cat([x, c], dim=1)
        return self.model(x).squeeze()

def main():
    args = get_args()
    source_folder_path: os.PathLike = args.source
    image_sizes: Tuple[int, int] = args.image_size
    batch_size: int = args.batch_size
    seed: int | None = args.seed
    z_size: int = args.z
    learning_rate: float = args.learning_rate
    epochs: int = args.epochs

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator_layer_size: List[int] = [256, 512, 1024]
    discriminator_layer_size: List[int] = [1024, 512, 256]

    if seed != None:
        torch.manual_seed(seed) 

    #   Resize images and normalize pixel values to [0, 1].
    transform: torchvision.transforms.Compose = transforms.Compose([
        transforms.Resize(image_sizes), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    dataset: datasets.ImageFolder = datasets.ImageFolder(source_folder_path, transform)

    #   Using 80-20 train-test split
    train_length: int = int(0.8 * len(dataset))         #   80%
    remaining_length: int = len(dataset) - train_length #   20%   
    validation_length: int = remaining_length // 2      #   10%
    test_length = remaining_length - validation_length  #   10%

    train_dataset: torch.utils.data.Subset
    test_dataset: torch.utils.data.Subset
    validation_dataset: torch.utils.data.Subset

    train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_length, test_length, validation_length])

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size)
    validation_loader: DataLoader = DataLoader(validation_dataset, batch_size=batch_size)

    classes: List[str] = dataset.classes
    class_count: int = len(classes)

    # for images, labels in validation_loader:        
    #     fig, ax = plt.subplots(figsize=(8, 4))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.imshow(images[0].permute(1, 2, 0).cpu())  # [H, W, C]
    #     plt.show()

    # Define generator
    generator: Generator = Generator(generator_layer_size, z_size, (3, *image_sizes), class_count).to(device)
    # Define discriminator
    discriminator: Discriminator = Discriminator(discriminator_layer_size, (3, *image_sizes), class_count).to(device)

    starting_epoch: int = 0

    if args.checkpoint != None:
        logger.info(f"Loading checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint["G"])
        discriminator.load_state_dict(checkpoint["D"])

        match = re.search(r"checkpoint_epoch_(\d+)\.pth", args.checkpoint)
        if match:
            starting_epoch = int(match.group(1))
        else:
            logger.info("Make sure checkpoint file name is in the format 'checkpoint_epoch_x.pth'.")
            exit(1)

    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    generator_optimizer: torch.optim.Adam = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    discriminator_optimizer: torch.optim.Adam = torch.optim.Adam(discriminator.parameters(), lr=2 * learning_rate)

    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0

    def generator_train_step(
            batch_size: int, 
            discriminator: nn.Module, 
            generator: nn.Module, 
            g_optimizer: torch.optim.Optimizer, 
            criterion: nn.BCEWithLogitsLoss
    ):
        # Init gradient
        g_optimizer.zero_grad()
        
        # Building z
        z = torch.randn(batch_size, z_size, device=device)
        
        # Building fake labels
        fake_labels = torch.randint(0, class_count, (batch_size,), device=device)
        
        # Generating fake images
        fake_images = generator(z, fake_labels)
        
        # Disciminating fake images
        validity = discriminator(fake_images, fake_labels)
        
        # Calculating discrimination loss (fake images)
        g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
        
        # Backword propagation
        g_loss.backward()
        
        #  Optimizing generator
        g_optimizer.step()
        
        return g_loss.item()

    def discriminator_train_step(
        batch_size: int, 
        discriminator: Discriminator, 
        generator: Generator, 
        d_optimizer: torch.optim.Optimizer, 
        criterion: nn.BCEWithLogitsLoss, 
        real_images: Tensor, 
        labels: List[str]
    ):  
        # Init gradient 
        d_optimizer.zero_grad()

        # Disciminating real images
        real_validity = discriminator(real_images, labels)
        
        # Calculating discrimination loss (real images)
        real_loss = criterion(real_validity, torch.ones(batch_size, device=device))
        
        # Building z
        z = torch.randn(batch_size, z_size, device=device)
        
        # Building fake labels
        fake_labels = torch.LongTensor(np.random.randint(0, class_count, batch_size), device=device)
        
        # Generating fake images
        fake_images = generator(z, fake_labels).detach()
        
        # Disciminating fake images
        fake_validity = discriminator(fake_images, fake_labels)
        
        # Calculating discrimination loss (fake images)
        fake_loss = criterion(fake_validity, torch.zeros(batch_size, device=device))
        
        # Sum two losses
        d_loss = real_loss + fake_loss
        
        # Backword propagation
        d_loss.backward()
        
        # Optimizing discriminator
        d_optimizer.step()
        
        return d_loss.item()

    def evaluate_discriminator_loss(
        discriminator: Discriminator,
        generator: Generator,
        criterion: nn.BCELoss,
        validation_loader: DataLoader
    ) -> float:
        discriminator.eval()
        generator.eval()
        total_loss = 0.0
        with torch.no_grad():
            for real_images, labels in validation_loader:
                real_images = real_images.to(device)
                labels = labels.to(device)
                batch_size = real_images.size(0)

                # Real loss
                real_validity = discriminator(real_images, labels)
                real_loss = criterion(real_validity, torch.ones(batch_size, device=device))

                # Fake loss
                z = torch.randn(batch_size, z_size, device=device)
                fake_labels = torch.randint(0, class_count, (batch_size,), device=device)
                fake_images = generator(z, fake_labels)
                fake_validity = discriminator(fake_images, fake_labels)
                fake_loss = criterion(fake_validity, torch.zeros(batch_size, device=device))

                total_loss += (real_loss + fake_loss).item()
        
        return total_loss / len(validation_loader)

    logger.info(f"Patience: {patience}")

    for epoch in range(starting_epoch, starting_epoch + epochs):
        logger.info("Epoch {}".format(epoch + 1))
        
        for i, (images, labels) in enumerate(train_loader):
            # Train data
            real_images = images.to(device)
            labels = labels.to(device)
            
            #   decaying random noise to to images
            noise = torch.randn_like(real_images) * max(0, 0.1*(1 - epoch/epochs))

            # Set generator train
            generator.train()
            
            # Train discriminator
            d_loss = discriminator_train_step(len(real_images), discriminator, generator, discriminator_optimizer, 
                criterion, real_images + noise, labels
            )
            
            # Train generator
            g_loss = generator_train_step(len(real_images), discriminator, generator, generator_optimizer, criterion)
        
        # Set generator eval
        generator.eval()
        
        logger.info(f'Generator Loss: {g_loss}, Discriminator Loss: {d_loss}')
        
        # Visualize & checkpoint
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                z = torch.randn(class_count, z_size, device=device)
                labels = torch.arange(class_count, device=device)
                sample_images = generator(z, labels)
                grid = make_grid(sample_images, nrow=class_count//2, normalize=True)
                plt.imshow(grid.permute(1,2,0).cpu()); plt.axis('off'); plt.show()
            torch.save({
                'G': generator.state_dict(),
                'D': discriminator.state_dict(),
                'g_opt': generator_optimizer.state_dict(),
                'd_opt': discriminator_optimizer.state_dict(),
            }, f"checkpoint_epoch_{epoch+1}.pth")
        
        val_d_loss = evaluate_discriminator_loss(discriminator, generator, criterion, validation_loader)
        logger.info(f"Validation Discriminator Loss: {val_d_loss:.4f}")

        if val_d_loss < best_val_loss:
            best_val_loss = val_d_loss
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs.")
            if patience_counter >= patience:
                logger.info("Early stopping due to lack of improvement in validation loss.")
                break
        
    torch.save({
        'G': generator.state_dict(),
        'D': discriminator.state_dict(),
        'g_opt': generator.state_dict(),
        'd_opt': discriminator.state_dict(),
    }, f"checkpoint_epoch_{epoch+1}.pth")
    # torch.save(generator.state_dict(), "generator")
    # torch.save(discriminator.state_dict(), "discriminator")

    test_loss = evaluate_discriminator_loss(discriminator, generator, criterion, test_loader)
    logger.info(f"Final testing loss: {test_loss}")

if __name__ == "__main__":
    exit(main())