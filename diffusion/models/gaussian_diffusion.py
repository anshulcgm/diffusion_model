import pdb
from typing import Optional

import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):
    def __init__(
        self, n_timesteps: int, variance_lower_bound: float = 0.001, variance_upper_bound: float = 0.002
    ) -> None:
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

        self.alphas = 1.0 - self.betas
        self.cumulative_alphas = torch.cumprod(self.alphas, axis=0)

    def add_noise(self, batch: torch.Tensor, timesteps: torch.Tensor, random_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adds gaussian noise to each image in the batch depending on the timestep

        Args:
            batch: Tensor of shape (b, c, h, w) representing images
            timesteps: Tensor of shape (b,) representing which timestep each batch is in
            noise: Tensor of shape (b, 1, 1, 1) representing random noise for each batch

        Returns:
            noised_batch: Tensor of shape (b, c, h, w) representing noise added to batch
        """
        b, _, _, _ = batch.shape
        curr_alphas = self.cumulative_alphas[timesteps].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        curr_variances = 1.0 - curr_alphas
        if random_noise is None:
            random_noise = torch.randn(size=[b, 1, 1, 1])
        noised_batch = torch.sqrt(curr_alphas) * batch + torch.sqrt(curr_variances) * random_noise
        return noised_batch
