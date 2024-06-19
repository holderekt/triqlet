"""
Filename: triqlet/triplet/distances.py
Author: Ivan Diliso
License: MIT License

This software is licensed under the MIT License.
"""


import torch.nn as nn
import torch


class EuclideanDistance(nn.Module):
    def __init__(self):
        super(EuclideanDistance, self).__init__()
    
    def forward(self, anchor, sample):
        return torch.norm(anchor - sample, dim=1, keepdim=True)


class LearnedDistance(nn.Module):
    def __init__(self, embedding_size, composer_function):

        self.composer_function = composer_function


        super(LearnedDistance, self).__init__()
        self.learner = nn.Sequential(
            nn.Linear(embedding_size, 1)
        )

    def forward(self, anchor, sample):
        return self.learner(self.composer_function((anchor, sample)))
