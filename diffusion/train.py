import pdb

import torch
from torch.nn import MSELoss, DataParallel

from diffusion.models.gaussian_diffusion import GaussianDiffusion
from diffusion.models.denoising_model import EncoderDecoder
from diffusion.data_utils import prep_data

NUM_TIMESTEPS = 30
TIME_EMB_DIM = 64
NUM_EPOCHS = 20

TRAINING_DIR = "archive/car_data/car_data/train"
BATCH_SIZE = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_loss(
    diffusion_model: GaussianDiffusion,
    denoising_model: EncoderDecoder,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    """Calculates difference between predicted noise by denoising model and inputted noise into gaussian model

    Args:
        diffusion_model: noising model
        denoising_model: encoder-decoder CNN that predicts noise given timesteps
        x_start: Starting unnoised images
        timesteps: Which part of the noising process each image in the batch is at
        criterion: Loss module to calculate loss between predicted noise and actual noise
    Returns:
        loss: Difference between predicted noise and inputted noise as estimated by criterion
    """
    noise = torch.randn_like(x_start)
    noised_images = diffusion_model.add_noise(batch=x_start, timesteps=timesteps, random_noise=noise)
    pred_noise = denoising_model(batch=noised_images.to(device), timesteps=timesteps.to(device))
    loss = criterion(pred_noise, noise.to(device))
    return loss

def train() -> None:
    """Training Loop for denoising model"""
    diffusion_model = GaussianDiffusion(n_timesteps = NUM_TIMESTEPS)
    denoising_model = EncoderDecoder(n_timesteps = NUM_TIMESTEPS, time_emb_dim = TIME_EMB_DIM)
    denoising_model = DataParallel(denoising_model, device_ids = [0, 1, 2, 3])
    denoising_model.to(device)
    training_dataloader = prep_data(train_dir = TRAINING_DIR, batch_size = BATCH_SIZE)
    print(f"There are {len(training_dataloader)} batches in the loader")
    criterion = MSELoss()
    optimizer = torch.optim.Adam(denoising_model.parameters(), lr = 0.001)
    for i in range(NUM_EPOCHS):
        total_loss = 0.0
        for j, (images, _) in enumerate(training_dataloader):
            optimizer.zero_grad()
            timesteps = torch.randint(low = 1, high = NUM_TIMESTEPS, size = [BATCH_SIZE])
            loss = calculate_loss(diffusion_model, denoising_model, images, timesteps, criterion)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            print(f"For epoch {i + 1}, batch {j + 1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(training_dataloader)
        print(f"For epoch {i + 1}: Average Loss = {avg_loss:.4f}")
    
    torch.save(denoising_model, "trained_denoising_model.pth")

if __name__ == "__main__":
    train()


