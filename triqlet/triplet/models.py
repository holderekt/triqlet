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
    def __init__(self, embedding : nn.Module , distance : nn.Module):
        super(TripletNet, self).__init__()
        self.embedding = embedding
        self.distance = distance

    def forward(self, anchor, positive, negative):

        a = self.embedding(anchor)
        p = self.embedding(positive)
        n = self.embedding(negative)

        dp = self.distance(a, p)
        dn = self.distance(a, n)

        return dp, dn
    

