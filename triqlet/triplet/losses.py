"""
Filename: triqlet/triplet/losses.py
Author: Ivan Diliso
License: MIT License

This software is licensed under the MIT License.
"""


import torch.nn as nn
import torch


class TripletLoss(nn.Module):
    """ Main triplet learning loss using distances of anchor example from positive and negative examples
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, dist_pos, dist_neg):
        loss = self.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()