import random
import pdb

import torch

from diffusion.models.gaussian_diffusion import GaussianDiffusion

random.seed(42)
torch.manual_seed(42)

def test_init():
    num_timesteps = random.randint(10, 30)
    model = GaussianDiffusion(num_timesteps)

def test_add_noise():
    num_timesteps = random.randint(10, 30)
    batch_size = random.randint(4, 10)
    model = GaussianDiffusion(num_timesteps)
    random_img_tensor = torch.rand(size = [batch_size, 3, 224, 224])
    curr_timesteps = torch.randint(low = 0, high = num_timesteps, size = [batch_size])
    noised_batch = model.add_noise(random_img_tensor, curr_timesteps)
    assert noised_batch.shape == random_img_tensor.shape
