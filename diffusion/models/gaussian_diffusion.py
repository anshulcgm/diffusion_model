import pdb
from typing import Optional

import torch


class GaussianDiffusion:
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

    def unsqueeze_multiple_dimensions(
        self, start_tensor: torch.Tensor, num_dimensions_to_unsqueeze: int = 3
    ) -> torch.Tensor:
        """Helper method to unsqueeze 1D tensors into n-dimensional tensors

        Args:
            start_tensor: 1D tensor of shape (n,)
            num_dimensions_unsqueeze: how many dimensions to add

        Returns:
            unsqueezed_tensor: tensor of shape (n, 1, 1, .....)
        """
        unsqueezed_tensor = start_tensor
        for i in range(num_dimensions_to_unsqueeze):
            unsqueezed_tensor = unsqueezed_tensor.unsqueeze(i + 1)
        return unsqueezed_tensor

    def add_noise(
        self, batch: torch.Tensor, timesteps: torch.Tensor, random_noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Adds gaussian noise to each image in the batch depending on the timestep

        Args:
            batch: Tensor of shape (b, c, h, w) representing images
            timesteps: Tensor of shape (b,) representing which timestep each batch is in
            noise: Tensor of shape (b, c, h, w) representing random noise

        Returns:
            noised_batch: Tensor of shape (b, c, h, w) representing noise added to batch
        """
        b, _, _, _ = batch.shape
        curr_cumulative_alphas = self.unsqueeze_multiple_dimensions(self.cumulative_alphas[timesteps])
        curr_cumulative_variances = 1.0 - curr_cumulative_alphas
        if random_noise is None:
            random_noise = torch.randn_like(batch)
        noised_batch = torch.sqrt(curr_cumulative_alphas) * batch + torch.sqrt(curr_cumulative_variances) * random_noise
        return noised_batch

    def calculate_starting_image(
        self, curr_images: torch.Tensor, curr_timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Calculates x_0 from x_t

        Args:
            curr_images: Tensor of shape (b, c, h, w) representing noised images
            curr_timesteps: Tensor of shape (b,) representing which timestep each element in the batch is at
            noise: Tensor of shape (b, c, h, w) indicating noise predictions by decoder network

        Returns:
            starting_images: Images before any gaussian noise was added
        """
        curr_cumulative_alphas = self.unsqueeze_multiple_dimensions(self.cumulative_alphas[curr_timesteps])
        curr_cumulative_variances = 1.0 - curr_cumulative_alphas
        starting_images = (
            1.0 / torch.sqrt(curr_cumulative_alphas) * curr_images
            - torch.sqrt(curr_cumulative_variances / curr_cumulative_alphas) * noise
        )
        return starting_images

    def calculate_posterior_mean(
        self, curr_images: torch.Tensor, curr_timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Calculates posterior mean for distribution that is used to sample x_{t - 1}

        Args:
            curr_images: Tensor of shape (b, c, h, w) representing noised images
            curr_timesteps: Tensor of shape (b,) representing which timestep each element in the batch is at
            noise: Tensor of shape (b, c, h, w) indicating noise predictions by decoder network

        Returns:
            posterior_mean: Tensor of shape (b, c, h, w) representing mean of distribution used to sample x_{t - 1}
        """
        starting_images = self.calculate_starting_image(curr_images, curr_timesteps, noise)
        previous_timestep_cumulative_alphas = self.unsqueeze_multiple_dimensions(
            self.cumulative_alphas[curr_timesteps - 1]
        )
        cumulative_alphas = self.unsqueeze_multiple_dimensions(self.cumulative_alphas[curr_timesteps])
        curr_alphas = self.unsqueeze_multiple_dimensions(self.alphas[curr_timesteps])
        curr_betas = self.unsqueeze_multiple_dimensions(self.betas[curr_timesteps])
        posterior_mean_first_coefficient = (
            torch.sqrt(previous_timestep_cumulative_alphas) * curr_betas / (1.0 - cumulative_alphas)
        )
        posterior_mean_second_coefficient = (
            torch.sqrt(curr_alphas) * (1.0 - previous_timestep_cumulative_alphas) / (1.0 - cumulative_alphas)
        )
        posterior_mean = (
            posterior_mean_first_coefficient * starting_images + posterior_mean_second_coefficient * curr_images
        )
        return posterior_mean

    def calculate_posterior_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Calculates variance of posterior distribution used to sample x_{t - 1}

        Args:
            timesteps: Tensor of shape (b,) representing which timestep each element in the batch is at

        Returns:
            posterior_variances: Tensor of shape (b,) representing a variance for each batch
            clipped_log_posterior_variance: Tensor of shape (b,) representing log of variances for numerical stability
        """
        previous_timestep_cumulative_alphas = self.cumulative_alphas[timesteps - 1]
        cumulative_alphas = self.cumulative_alphas[timesteps]
        curr_betas = self.betas[timesteps]
        posterior_variance = (1.0 - previous_timestep_cumulative_alphas) / (1.0 - cumulative_alphas) * curr_betas
        clipped_log_posterior_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        return posterior_variance, clipped_log_posterior_variance

    def calculate_previous_timestep_images(
        self, curr_images: torch.Tensor, curr_timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Calculates x_{t - 1} from x_{t}

        Args:
            curr_images: Tensor of shape (b,c,h,w) that represents noised images at current timesteps
            curr_timesteps: Tensor of shape (b,) that represents which timestep each element is at
            noise: Predicted noise from previous timestep

        Returns:
            x_{t - 1}: Predicted images from previous timestep
        """
        posterior_mean = self.calculate_posterior_mean(curr_images, curr_timesteps, noise)
        _, log_variance = self.calculate_posterior_variance(curr_timesteps)
        backward_noise = torch.randn_like(curr_images)
        zero_timestep_mask = curr_timesteps == 0
        previous_timestep_images = (
            posterior_mean + self.unsqueeze_multiple_dimensions(torch.exp(0.5 * log_variance)) * backward_noise
        )
        previous_timestep_images[zero_timestep_mask] = curr_images[zero_timestep_mask]
        return previous_timestep_images
