"""
Filename: triqlet/triplet/embedders.py
Author: Ivan Diliso
License: MIT License

This software is licensed under the MIT License.
"""

import torch.nn as nn
import torch
from ..utils import RotationScaler



class CNNAnglesEmbeddingNet(nn.Module):
    """ Simple CNN to embedd an image to a specified size vector, expects an images in the format
    (1x28x28) with only one single color channel. The ouput vector will be normalized in the range [0,2pi] for a feasible input 
    to quantum neural networks with angles encodings.
    """
    def __init__(self, embedding_size : int):
        """Initiate class with an output embedding size 

        Args:
            embedding_size (int): Ouput embedding size 
        """
        super(CNNAnglesEmbeddingNet, self).__init__()

        self.embedding_size = embedding_size

        # Main convolution blocks
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5)),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,5)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        # Tensor dimensionality reduction

        self.reduction = nn.Sequential(
            nn.Linear(in_features=576, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=embedding_size)
        )

        # Scaling for quantum encoding networks

        self.scaler = RotationScaler(scale=2*torch.pi)

    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        x = self.flatten(x)
        x = self.reduction(x)
        
        x = self.scaler(x)

        return x
    

