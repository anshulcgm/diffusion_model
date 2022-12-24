import pdb

import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):
    def __init__(self, n_timesteps: int, variance_lower_bound: float = 0.001, variance_upper_bound: float = 0.002) -> None:
        """Define variance schedule along with important constants for calculations

        Args:
            n_timesteps: Number of timesteps in forward corruption process
        """

        self.n_timesteps = n_timesteps
        self.variance_lower_bound = variance_lower_bound
        self.variance_upper_bound = variance_upper_bound

        self.variance_start = 1000 / self.n_timesteps * self.variance_lower_bound
        self.variance_end = 1000 / self.n_timesteps * self.variance_upper_bound
        self.betas = torch.linspace(self.variance_start, self.variance_end, self.n_timesteps)

        self.alphas = 1. - self.betas
        self.cumulative_alphas = torch.cumprod(self.alphas, axis = 0)
    
    def add_noise(self, batch: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Adds gaussian noise to each image in the batch depending on the timestep

        Args:
            batch: Tensor of shape (b, c, h, w) representing images
            timesteps: Tensor of shape (b,) representing which timestep each batch is in
        
        Returns:
            noised_batch: Tensor of shape (b, c, h, w) representing noise added to batch
        """
        b, _, _, _ = batch.shape
        curr_alphas = self.alphas[timesteps].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        curr_betas = self.betas[timesteps].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        random_noise = torch.randn(size = [b, 1, 1, 1])
        noised_batch = torch.sqrt(curr_alphas) * batch + torch.sqrt(curr_betas) * random_noise
        return noised_batch