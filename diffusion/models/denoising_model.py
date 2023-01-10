from typing import Tuple
import pdb

import torch
import torch.nn as nn
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
