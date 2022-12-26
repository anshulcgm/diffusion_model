from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

class DenoisingModel(nn.Module):
    def __init__(self, n_timesteps: int, time_emb_dim: int, img_dim: int) -> None:
        """Initializes necessary layers for embeddings and convolutions 

        Args:
            n_timesteps: How many timesteps images are noised over
            time_emb_dim: Embedding dimension for time tokens
            img_dim: Initial dimension of image tokens
        """ 
        super(DenoisingModel, self).__init__()
        self.time_embedder = nn.Embedding(n_timesteps, time_emb_dim)
        self.encoder = models.resnet18(pretrained = False)
        
        