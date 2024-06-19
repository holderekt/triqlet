"""
Filename: triqlet/quantum/models.py
Author: Ivan Diliso
License: MIT License

This software is licensed under the MIT License.
"""


from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from abc import ABC, abstractmethod
from .encoders import *
import numpy as np
from typing import List
from torch import nn
import torch
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from IPython.display import display
from .layers import QuantumLayer


class QuantumSamplerModel(nn.Module):
    """Modular quantum torch layer based on sampling measurements
    """
    def __init__(self, circ_qubits : int, encoder : QuantumCircuit , ansatz : QuantumCircuit, shots : int, measurement : List[int]):
        """Create a QuantumCircuit with sampling mesurement based on classical data circuit encoding and a variational quantum circuit
        with learnable parameters

        Args:
            circ_qubits (int): Number of qubits in circuit
            encoder (QuantumCircuit): Classical data encoder circuit
            ansatz (QuantumCircuit): Variational quantum circuit with learnable parameters
            shots (int): Number of measurements
            measurement (List[int]): Index of measured qubits
        """
        super(QuantumSamplerModel, self).__init__()

        self.encoder = encoder
        self.circ_qubits = circ_qubits
        self.ansatz = ansatz

        self.args = {
            "shots" : shots
        }

        self.output_size = len(measurement)**2

        self.qr = QuantumRegister(circ_qubits, "q")
        self.cr = ClassicalRegister(len(measurement), "meas")

        self.qnn = QuantumCircuit(self.qr, self.cr)
        self.qnn.compose(encoder, inplace=True)
        self.qnn.compose(ansatz, inplace=True)

        for i,qb in enumerate(measurement):
            self.qnn.measure(qb,i)

        self.sampler = Sampler(options=self.args)

        self.qnn_net = SamplerQNN(
            sampler=self.sampler,
            circuit = self.qnn,
            input_params=self.encoder.parameters,
            weight_params=self.ansatz.parameters,
            input_gradients=True
        )

        self.quantum_layer = TorchConnector(self.qnn_net)


    def forward(self, anchor : torch.Tensor) -> torch.Tensor:
        """Compute QNN ouput using sampling measurements

        Args:
            anchor (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        return self.quantum_layer(anchor)[:, :self.output_size]
    
    
    def draw(self, output="mpl", decompose=True):
        """Quantum circuit visualizazione

        Args:
            output (str, optional): Visualization backend. Defaults to "mpl".
            decompose (bool, optional): Visualize layes gate (True) or block representation (False). Defaults to True.
        """
        if decompose:
            display(self.qnn.decompose().draw(output))
        else:
            display(self.qnn.draw(output))




