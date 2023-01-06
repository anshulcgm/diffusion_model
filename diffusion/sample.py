import pdb
import os

import torch
import torchvision.transforms as T

from PIL import Image

from diffusion.models.gaussian_diffusion import GaussianDiffusion
from diffusion.models.denoising_model import DenoisingModel

NUM_TIMESTEPS = 30
TIME_EMB_DIM = 64

BATCH_SIZE = 4
IMG_SIZE = 224

SAVED_MODEL_PATH = "trained_denoising_model.pth"

def sample_from_random_noise(diffusion_model: GaussianDiffusion, denoising_model: DenoisingModel) -> torch.Tensor:
    """Samples x_0 from x_T

    Args:
        diffusion_model: Helper model with mathematical functions for noising images and reversing noising process
        denoising_model: Torch model that predicts the noise given a timestep and noised images
    
    Returns:
        x_0: Images after denoising process is complete
    """

    random_noise = torch.randn(size = [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE])
    timesteps = torch.ones(size = [BATCH_SIZE]).to(torch.int64) * NUM_TIMESTEPS - 1
    curr_images = random_noise
    for i in range(NUM_TIMESTEPS - 1, 0, -1):
        pred_noise = denoising_model(curr_images, timesteps)
        curr_images = diffusion_model.calculate_previous_timestep_images(curr_images, timesteps, pred_noise)
        timesteps -= 1
    
    return curr_images

def save_images(images: torch.Tensor) -> None:
    """Helper method for saving torch tensors as images

    Args
        images: Tensor of shape (b, c, h, w)
    """
    pil_transform = T.ToPILImage()
    b, _, _, _ = images.shape
    for i in range(b):
        curr_img = pil_transform(images[i])
        curr_img.save(f"{i}.png")

def main():
    pdb.set_trace()
    diffusion_model = GaussianDiffusion(n_timesteps = NUM_TIMESTEPS)
    denoising_model = DenoisingModel(n_timesteps = NUM_TIMESTEPS, time_emb_dim = TIME_EMB_DIM)
    if os.path.exists(SAVED_MODEL_PATH):
        denoising_model.load_state_dict(torch.load(SAVED_MODEL_PATH).module.state_dict())
    sampled_images = sample_from_random_noise(diffusion_model, denoising_model)
    sampled_images.clamp_(0., 1.)
    unnormalized_images = (sampled_images + 1) * 0.5
    save_images(unnormalized_images)
    
if __name__ == "__main__":
    main()