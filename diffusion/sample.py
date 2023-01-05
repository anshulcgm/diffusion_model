import pdb

import torch

from diffusion.models.gaussian_diffusion import GaussianDiffusion
from diffusion.models.denoising_model import DenoisingModel

NUM_TIMESTEPS = 20
TIME_EMB_DIM = 64

BATCH_SIZE = 4
IMG_SIZE = 224

def sample_from_random_noise(diffusion_model: GaussianDiffusion, denoising_model: DenoisingModel) -> torch.Tensor:
    random_noise = torch.randn(size = [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE])
    timesteps = torch.ones(size = [BATCH_SIZE]) * NUM_TIMESTEPS - 1
    curr_images = random_noise
    for i in range(NUM_TIMESTEPS - 1, 0, -1):
        pred_noise = denoising_model(curr_images, timesteps)
        curr_images = diffusion_model.calculate_previous_timestep_images(curr_images, timesteps, pred_noise)
        timesteps -= 1
    
    return curr_images

def main():
    diffusion_model = GaussianDiffusion(n_timesteps = NUM_TIMESTEPS)
    denoising_model = DenoisingModel(n_timesteps = NUM_TIMESTEPS, time_emb_dim = TIME_EMB_DIM)
    samples_images = sample_from_random_noise(diffusion_model, denoising_model)

if __name__ == "__main__":
    main()