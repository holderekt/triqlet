from qiskit import QuantumCircuit, QuantumRegister
from sklearn import datasets
import matplotlib.pyplot as plt
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics import silhouette_score
import sklearn.metrics.cluster as cluster_metrics
import numpy as np
from os import name
from qiskit.circuit import Parameter, ParameterVector
from qiskit_machine_learning.circuit.library import RawFeatureVector
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from IPython.display import clear_output
from qiskit_machine_learning.algorithms.regressors import VQR, NeuralNetworkRegressor
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, ADAM, GradientDescent, QNSPSA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import DistanceMetric
import pickle
from sklearn.cluster import AgglomerativeClustering, KMeans
from PIL import Image
from qiskit_machine_learning.utils.loss_functions import L1Loss, L2Loss, KernelLoss
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVR
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
import torch
from torch import nn
from qiskit_machine_learning.connectors import TorchConnector
from torch.optim import Adam
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from torchvision import models
from torchsummary import summary
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchsummary import summary
from PIL import Image
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from torchvision import datasets
from sklearn.decomposition import PCA
from qiskit import ClassicalRegister
from qiskit.primitives import Sampler, BaseSamplerV1