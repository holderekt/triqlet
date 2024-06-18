from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from abc import ABC, abstractmethod
from .encoders import *
import numpy as np
from typing import List
from torch import nn
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from IPython.display import display




class QuantumSamplerModel(nn.Module):
    def __init__(self, circ_qubits, encoder, ansatz, shots, measurement):
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


    def forward(self, anchor):
        return self.quantum_layer(anchor)[:, :self.output_size]
    
    
    def draw(self, output="mpl", decompose=True):
        if decompose:
            display(self.qnn.decompose().draw(output))
        else:
            display(self.qnn.draw(output))




