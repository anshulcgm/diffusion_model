import random
import pdb

import torch

from diffusion.models.gaussian_diffusion import GaussianDiffusion

random.seed(42)
torch.manual_seed(42)


def test_init():
    num_timesteps = random.randint(10, 30)
    model = GaussianDiffusion(num_timesteps)


def test_add_noise_shape():
    num_timesteps = random.randint(10, 30)
    batch_size = random.randint(4, 10)
    model = GaussianDiffusion(num_timesteps)
    random_img_tensor = torch.rand(size=[batch_size, 3, 224, 224])
    curr_timesteps = torch.randint(low=0, high=num_timesteps, size=[batch_size])
    noised_batch = model.add_noise(random_img_tensor, curr_timesteps)
    assert noised_batch.shape == random_img_tensor.shape

def test_add_noise_specific_values():
    img_tensor = torch.randn(size = [4, 3, 2, 2])
    random_noise = torch.randn(size = [4, 1, 1, 1])
    noised_batch = torch.zeros_like(img_tensor)
    model = GaussianDiffusion(n_timesteps = 20)
    timesteps = torch.Tensor([4, 3, 10, 9]).to(torch.int64)
    for i in range(img_tensor.shape[0]):
        curr_img = img_tensor[i]
        curr_timestep = timesteps[i]
        curr_noise = random_noise[i].item()
        alpha_value = model.cumulative_alphas[curr_timestep]
        variance = 1. - alpha_value
        noisy_img = torch.sqrt(alpha_value) * curr_img + torch.sqrt(variance) * curr_noise
        noised_batch[i] = noisy_img
    model_output = model.add_noise(img_tensor, timesteps, random_noise)
    assert torch.equal(model_output, noised_batch)