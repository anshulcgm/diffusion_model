from typing import Tuple
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
import torchvision.models as models


class EncoderDecoder(nn.Module):
    def __init__(self, n_timesteps: int, time_emb_dim: int) -> None:
        """Initializes necessary layers for embeddings and convolutions

        Args:
            n_timesteps: How many timesteps images are noised over
            time_emb_dim: Embedding dimension for time tokens
            img_dim: Initial dimension of image tokens
        """
        super(EncoderDecoder, self).__init__()

        self.time_emb_dim = time_emb_dim
        self.time_embedder = nn.Embedding(n_timesteps, time_emb_dim)
        self.encoder = models.resnet18(pretrained=False)
        self.upsample_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.upsample_3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1)

        self.activation = nn.ReLU()

    def forward_encoder(self, batch: torch.Tensor) -> torch.Tensor:
        """Passes batch through first 2 layers of resnet

        Args:
            batch: Tensor of shape (b, c, h, w)

        Returns:
            encoded_batch: Tensor of shape (b, 128, h / 8, w / 8)
        """
        x = self.encoder.conv1(batch)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        encoded_batch = self.encoder.layer2(x)

        return encoded_batch

    def forward(self, batch: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predicts noise for each image in the batch

        Args:
            batch: Tensor of shape (b, c, h, w) representing batch of noised images
            timesteps: Tensor of shape (b,) representing timestep for each element in the batch

        Returns:
            noise: Tensor of shape (b, c, h, w) representing predicted noise for each image
        """
        b, c, h, w = batch.shape
        assert (h * w) % self.time_emb_dim == 0  # Needs to be true to inject time embeddings
        time_embeddings = self.time_embedder(timesteps)
        time_embeddings = time_embeddings.reshape(b, 1, self.time_emb_dim, 1)
        x = batch.reshape(b, c, self.time_emb_dim, -1)
        x = x + time_embeddings
        x = x.reshape(b, c, h, w)
        x = self.forward_encoder(x)
        x = self.activation(self.upsample_1(x))
        x = self.activation(self.upsample_2(x))
        x = self.upsample_3(x)
        return x


class Segnet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        """Initializes necessary layers for segnet forward pass

        Args:
            in_channels: How many channels will the input image have
            out_channels: How many channels we want the output image to have
        """
        super(Segnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize encoder layers
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder1 = self.initialize_encoder_layer(in_channels=in_channels, out_channels=64, num_extra_layers=1)
        self.encoder2 = self.initialize_encoder_layer(in_channels=64, out_channels=128, num_extra_layers=1)
        self.encoder3 = self.initialize_encoder_layer(in_channels=128, out_channels=256, num_extra_layers=2)
        self.encoder4 = self.initialize_encoder_layer(in_channels=256, out_channels=512, num_extra_layers=2)
        self.encoder5 = self.initialize_encoder_layer(in_channels=512, out_channels=512, num_extra_layers=2)
        self.encoder = ModuleList([self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5])

        # Initialize decoder layers
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder1 = self.initialize_decoder_layer(in_channels=512, out_channels=512, num_extra_layers=2)
        self.decoder2 = self.initialize_decoder_layer(in_channels=512, out_channels=256, num_extra_layers=2)
        self.decoder3 = self.initialize_decoder_layer(in_channels=256, out_channels=128, num_extra_layers=2)
        self.decoder4 = self.initialize_decoder_layer(in_channels=128, out_channels=64, num_extra_layers=1)
        self.decoder5 = self.initialize_decoder_layer(in_channels=64, out_channels=out_channels, num_extra_layers=1)
        self.decoder = ModuleList([self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5])

    def initialize_encoder_layer(self, in_channels: int, out_channels: int, num_extra_layers: int = 1) -> ModuleList:
        """Initializes an encoder segnet layer

        Args:
            in_channels: How many channels are passed in
            out_channels: How many channels we want to convolve the output to
            num_extra_layers: Number of channel preserving convolutions plus batch norms to add
        Returns:
            encoder_layer: Module list containing all convolutions and batch norms
        """
        module_list = []
        layer_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        layer_bn1 = nn.BatchNorm2d(num_features=out_channels)
        module_list.append(layer_conv1)
        module_list.append(layer_bn1)
        for _ in range(num_extra_layers):
            module_list.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1))
            module_list.append(nn.BatchNorm2d(num_features=out_channels))

        return ModuleList(module_list)

    def initialize_decoder_layer(self, in_channels: int, out_channels: int, num_extra_layers: int = 1) -> ModuleList:
        """Initializes a decoder segnet layer

        Args:
            in_channels: How many channels are passed in
            out_channels: How many channels we want to convolve the output to
            num_extra_layers: Number of channel preserving convolutions plus batch norms to add

        Returns:
            decoder_layer: Module list containing all convolutions and batch norms
        """
        module_list = []
        for _ in range(num_extra_layers):
            module_list.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1))
            module_list.append(nn.BatchNorm2d(num_features=in_channels))

        module_list.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
        module_list.append(nn.BatchNorm2d(num_features=out_channels))

        return ModuleList(module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes x through segnet network"""
        x_sizes = []
        max_pooling_indices = []

        for encoder_layer in self.encoder:
            conv1 = encoder_layer[0]
            bn1 = encoder_layer[1]
            x = F.relu(bn1(conv1(x)))
            num_remaining_layers = len(encoder_layer) - 2
            for j in range(0, num_remaining_layers, 2):
                conv_layer = encoder_layer[j]
                bn = encoder_layer[j + 1]
                x = F.relu(bn(conv_layer(x)))
            x, curr_pooling_indices = self.max_pool(input=x)
            x_sizes.append(x.size())
            max_pooling_indices.append(curr_pooling_indices)

        max_pooling_indices = max_pooling_indices[::-1]
        x_sizes = x_sizes[::-1]
        for i, decoder_layer in enumerate(self.decoder):
            x = self.max_unpool(input=x, indices=max_pooling_indices[i], output_size=x_sizes[i])
            num_extra_layers = len(decoder_layer) - 2
            for j in range(0, num_extra_layers, 2):
                conv_layer = decoder_layer[j]
                bn = decoder_layer[j + 1]
                x = F.relu(bn(conv_layer(x)))

            downsizing_conv = decoder_layer[-2]
            downsizing_bn = decoder_layer[-1]
            x = F.relu(downsizing_bn(downsizing_conv(x)))

        return x


if __name__ == "__main__":
    pdb.set_trace()
    segnet = Segnet(in_channels=3, out_channels=3)
    x = torch.randn(size=[4, 3, 224, 224])
    segnet(x)
