"""
Filename: triqlet/utils.py
Author: Ivan Diliso
License: MIT License

This software is licensed under the MIT License.
"""

import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


class RotationScaler(nn.Module):
    def __init__(self, scale=1):
        super(RotationScaler, self).__init__()

        self.sig = torch.nn.Sigmoid()
        self.scale = scale

    def forward(self, x):
        x = self.sig(x)
        x = x * self.scale
        
        return x

def triple_train(model, epochs, optimizer, criterion, train_data_loader, device, print_at=1):

    for epoch in range(epochs):

        prnt = (epoch % print_at) != 0 if epoch!=(epochs-1) else False

        if not(prnt):
            print(f"{color.BOLD}Epoch {color.END}{epoch+1}")

        ### --> Training Phase

        model.train()

        train_loss = 0.0
        train_samples = 0


        for anchor, positive, negative in tqdm(train_data_loader, disable=prnt):
            optimizer.zero_grad()

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            e_P, e_N = model(anchor, positive, negative)
            loss = criterion(e_P, e_N)

            loss.backward()
            optimizer.step()

            train_loss += loss 
            train_samples += positive.size(0)


        train_loss /= len(train_data_loader)
        
        if not(prnt):
            print(f"TRAINING   -> Loss: {color.RED}{train_loss:2.6f}{color.END}")
            print("")    


def embed(model, dataloader, embed_size, device):
    embedder = model.embedding
    with torch.no_grad():
        embedder.eval()
        embedding_data = np.empty((0, embed_size))
        for anchor, _,_ in tqdm(dataloader):
            anchor = anchor.to(device)
            embedding_data = np.concatenate((embedding_data, embedder(anchor).to("cpu").squeeze(0).numpy()), axis=0)
    return embedding_data


def automatic_backend_chooser():
    device = torch.device("cpu")

    if not torch.cuda.is_available():
        print(f"{color.GREEN}CUDA{color.END} NOT Available")
    else:
        print(f"{color.GREEN}CUDA{color.END} Available")
        device = torch.device("cuda")

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
    else:
        print(f"{color.BLUE}MPS{color.END} Available")
        device = torch.device("mps")
    print(f"Using {color.BOLD}{color.PURPLE}{str(device).upper()}{color.END} Acceleration")

    return device
