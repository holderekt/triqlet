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
    """_summary_
    """
    def __init__(self, data : np.array, target : np.array, num_classes : int, output_transform : transforms):
        """_summary_

        Args:
            data (np.array): _description_
            target (np.array): _description_
            num_classes (int): _description_
            output_transform (transforms): _description_
        """
        self.data = data
        self.target = target
        self.num_classes = num_classes
        self.output_transform = output_transform
        self.apply_transform = not(self.output_transform is None)
        self.len = len(self.target)
        self.classes = {i:(np.where(self.target == i)[0], np.where(self.target != i)[0] ) for i in range(num_classes)}


    def __getitem__(self, idx : int) -> torch.Tensor:
        """_summary_

        Args:
            idx (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.get_anchor(idx), self.get_positive(idx), self.get_negative(idx)
    

    def get_anchor(self, idx : int) -> torch.Tensor:
        """_summary_

        Args:
            idx (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.apply_transform:
            return self.output_transform(self.data[idx])
        else:
            return self.data[idx]
    

    def get_positive(self,idx : int) -> torch.Tensor:
        """_summary_

        Args:
            idx (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        i = np.random.choice(self.classes[self.target[idx]][0])
        if self.apply_transform:
            return self.output_transform(self.data[i])
        else:
            return self.data[idx]
    

    def get_negative(self,idx : int) -> torch.Tensor:
        """_summary_

        Args:
            idx (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        i = np.random.choice(self.classes[self.target[idx]][1])
        if self.apply_transform:
            return self.output_transform(self.data[i])
        else:
            return self.data[idx]


    def get_order(self) -> np.array:
        """_summary_

        Returns:
            np.array: _description_
        """
        return self.target.argsort()


    def get_flatten(self) -> np.array:
        """_summary_

        Returns:
            np.array: _description_
        """
        return self.data.reshape((self.data.shape[0], self.data.shape[1]**2))


    def __len__(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return self.len