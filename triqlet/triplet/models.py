"""
Filename: triqlet/triplet/models.py
Author: Ivan Diliso
License: MIT License

This software is licensed under the MIT License.
"""

import torch.nn as nn
import torch
import numpy as np


class TripletNet(nn.Module):
    """ Triplet learning net with siamese network structur main implementation.
    Expects and embedding model, able to embed data in a new format a and a distance model,
    that specifies the distances used for the triplet loss.
    """
    def __init__(self, embedding : nn.Module , distance : nn.Module):
        """ 
        Args:
            embedding (nn.Module): Embedding model (Data) -> (Data)
            distance (nn.Module): Distance model (Data, Data) -> (Distances)
        """
        super(TripletNet, self).__init__()
        self.embedding = embedding
        self.distance = distance

    def forward(self, anchor : torch.Tensor, positive : torch.Tensor, negative : torch.Tensor) -> torch.Tensor:
        """ Embed anchor, negative and positive examples and computer thier distances

        Args:
            anchor (torch.Tensor): Anchor examples
            positive (torch.Tensor): Example of the same class of anchor
            negative (torch.Tensor): Example of a different class of anchor

        Returns:
            torch.Tensor: Distance from positive and distance from negative (DisPositive, DisNegative)
        """
        a = self.embedding(anchor)
        p = self.embedding(positive)
        n = self.embedding(negative)

        dp = self.distance(a, p)
        dn = self.distance(a, n)

        return dp, dn
    

