"""
Filename: triqlet/triplet/distances.py
Author: Ivan Diliso
License: MIT License

This software is licensed under the MIT License.
"""


import torch.nn as nn
import torch
from collections.abc import Callable


class EuclideanDistance(nn.Module):
    """ Layer that computes euclidean distance from two input vector
    """
    def __init__(self):
        super(EuclideanDistance, self).__init__()
    
    def forward(self, anchor : torch.Tensor, sample : torch.Tensor) -> torch.Tensor:
        """Calculate euclidean distance between anchor and sample tensors

        Args:
            anchor (torch.Tensor): First tensor
            sample (torch.Tensor): Second tensor

        Returns:
            torch.Tensor: Unsqueezed tensor of euclidean distances between all vector (not pairwise)
        """
        return torch.norm(anchor - sample, dim=1, keepdim=True)



class LearnedDistance(nn.Module):
    """ Compute distances with with learnable parameters using a linear layer
    """
    def __init__(self, embedding_size : int, composer_function : Callable):
        """Define the LearnedDistance parameters specifying the embedding size and a composer function that will
        compose, stack or compress the two vector in a single tensor

        Args:
            embedding_size (int): Input size for the linear layer
            composer_function (Callable): Composer function for the two tensors (Tensor, Tensor) -> Tensor
        """
        self.composer_function = composer_function


        super(LearnedDistance, self).__init__()
        self.learner = nn.Sequential(
            nn.Linear(embedding_size, 1)
        )

    def forward(self, anchor, sample):
        return self.learner(self.composer_function((anchor, sample)))
