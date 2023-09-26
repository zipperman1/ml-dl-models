import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''

    # Реализуйте блок вида conv -> relu -> max_pooling. 
    # Параметры слоя conv заданы параметрами функции encoder_block. 
    # MaxPooling должен быть с ядром размера 2.
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )

    return block

def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
  
    # Реализуйте блок вида conv -> relu -> upsample.  
    # Параметры слоя conv заданы параметрами функции encoder_block. 
    # Upsample должен быть со scale_factor=2. Тип upsampling (mode) можно выбрать любым.
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear')

    )

    return block

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()


        # добавьте несколько слоев encoder block
        # это блоки-составляющие энкодер-части сети
        self.encoder = nn.Sequential(
            encoder_block(3, 32, 7, 3),
            encoder_block(32, 64, 3, 1),
            encoder_block(64, 128, 3, 1),
        )

        # добавьте несколько слоев decoder block
        # это блоки-составляющие декодер-части сети
        self.decoder = nn.Sequential(
            decoder_block(128, 64, 3, 1),
            decoder_block(64, 32, 3, 1),
            decoder_block(32, 3, 3, 1),
        )

    def forward(self, x):

        # downsampling 
        latent = self.encoder(x)

        # upsampling
        reconstruction = self.decoder(latent)

        return reconstruction
