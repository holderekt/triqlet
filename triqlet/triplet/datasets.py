"""
Filename: triqlet/triplet/datasets.py
Author: Ivan Diliso
License: MIT License

This software is licensed under the MIT License.
"""

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from abc import ABC, abstractmethod

class TripletDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def get_anchor(self, index):
        pass

    @abstractmethod
    def get_positive(self, index):
        pass

    @abstractmethod
    def get_negative(self, index):
        pass



class TripletImageDataset(TripletDataset):
    """Image data loader with ouput transform operator and random sampling for negative and positive examples.
    Produces a random triples consisting of (anchor, positive, negative), the class for the negative sample is chosen at random
    from a uniform probability distribution.
    """
    def __init__(self, data : np.array, target : np.array, num_classes : int, output_transform : transforms):
        """Creates calss from numpy data matrix (K,N,M) K training samples of images of size (MxN) 

        Args:
            data (np.array): Data matrix of size (K,N,M) K training samples of images of size (MxN) 
            target (np.array): Classes of each eaxample of size (K,1)
            num_classes (int): Number of different classes
            output_transform (transforms): Torch transformation applyed before output
        """
        self.data = data
        self.target = target
        self.num_classes = num_classes
        self.output_transform = output_transform
        self.apply_transform = not(self.output_transform is None)
        self.len = len(self.target)
        self.classes = {i:(np.where(self.target == i)[0], np.where(self.target != i)[0] ) for i in range(num_classes)}


    def __getitem__(self, idx : int) -> torch.Tensor:
        """Produce a triplet of (anchor, positive, negative) examples

        Args:
            idx (int): Index of anchor example

        Returns:
            torch.Tensor: Random sampling triples (anchor, positive, negative)
        """
        return self.get_anchor(idx), self.get_positive(idx), self.get_negative(idx)
    

    def get_anchor(self, idx : int) -> torch.Tensor:
        """Get anchor data 

        Args:
            idx (int): Index of anchor sample

        Returns:
            torch.Tensor: Transformed image
        """
        if self.apply_transform:
            return self.output_transform(self.data[idx])
        else:
            return self.data[idx]
    

    def get_positive(self,idx : int) -> torch.Tensor:
        """Generate an example from the same class of the anchor example

        Args:
            idx (int): Index of anchor sample

        Returns:
            torch.Tensor: Transformed image
        """
        i = np.random.choice(self.classes[self.target[idx]][0])
        if self.apply_transform:
            return self.output_transform(self.data[i])
        else:
            return self.data[idx]


    def get_negative(self,idx : int) -> torch.Tensor:
        """Generate an example from a different class of the anchor example chosen form uniform probability distribution
        among classes.

        Args:
            idx (int): Index of anchor sample

        Returns:
            torch.Tensor: Transformed image
        """
        i = np.random.choice(self.classes[self.target[idx]][1])
        if self.apply_transform:
            return self.output_transform(self.data[i])
        else:
            return self.data[idx]


    def get_order(self) -> np.array:
        """Sort examples by class

        Returns:
            np.array: Sorted examples
        """
        return self.target.argsort()


    def get_flatten(self) -> np.array:
        """Return dataset of flatten image examples

        Returns:
            np.array: Flatten dataset
        """
        return self.data.reshape((self.data.shape[0], self.data.shape[1]**2))


    def __len__(self) -> int:
        """Number of training examples

        Returns:
            int: Number of training examples
        """
        return self.len