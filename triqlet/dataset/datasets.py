import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class TripletDataset(Dataset):
    def __init__(self, data, target, num_classes, output_transform):
        self.data = data
        self.target = target
        self.num_classes = num_classes
        self.output_transform = output_transform
        self.len = len(self.target)
        self.classes = {i:(np.where(self.target == i)[0], np.where(self.target != i)[0] ) for i in range(num_classes)}


    def __getitem__(self, idx):
        return self.get_anchor(idx), self.get_positive(idx), self.get_negative(idx)
    

    def get_anchor(self, idx):
        return self.output_transform(self.data[idx])
    

    def get_positive(self, idx):
        i = np.random.choice(self.classes[self.target[idx]][0])
        return self.output_transform(self.data[i])
    

    def get_negative(self, idx):
        i = np.random.choice(self.classes[self.target[idx]][1])
        return self.output_transform(self.data[i])


    def get_order(self):
        return self.target.argsort()


    def ordered_pairwise(self):
        return self.kers[:,self.get_order()][self.get_order(),:]


    def get_flatten(self):
        return self.data.reshape((self.data.shape[0], self.data.shape[1]**2))


    def __len__(self):
        return self.len