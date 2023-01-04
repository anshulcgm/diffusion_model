import random
import pdb

import torch

from diffusion.models.denoising_model import DenoisingModel

random.seed(42)
torch.manual_seed(42)


def test_init():
    num_timesteps = random.randint(10, 30)
    denoising_model = DenoisingModel(n_timesteps=num_timesteps, time_emb_dim=64)


def test_forward_encoder():
    num_timesteps = random.randint(10, 30)
    denoising_model = DenoisingModel(n_timesteps=num_timesteps, time_emb_dim=64)
    height = 8 * random.randint(1, 50)
    width = 8 * random.randint(1, 50)
    batch = torch.randn(size=[4, 3, height, width])
    encoded_batch = denoising_model.forward_encoder(batch)
    assert encoded_batch.shape == torch.Size([4, 128, height // 8, width // 8])


def test_forward():
    num_timesteps = random.randint(10, 30)
    denoising_model = DenoisingModel(n_timesteps=num_timesteps, time_emb_dim=64)
    height = 8 * random.randint(1, 50)
    width = 8 * random.randint(1, 50)
    batch = torch.randn(size=[4, 3, height, width])
    timesteps = torch.randint(low=0, high=num_timesteps, size=[4])
    pred_noise = denoising_model(batch, timesteps)
    assert pred_noise.shape == batch.shape
