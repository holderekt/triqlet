from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from abc import ABC, abstractmethod
from .encoders import *
import numpy as np
from typing import List



class QuantumLayer(ABC):
    """ Abstract class for quantum layer definition
    """
    @abstractmethod
    def __new__(self) -> QuantumCircuit:
        return None



class EncoderLayer(QuantumLayer):
    """Create a quantum encoder circuit from encoder name
    """
    def __new__(self, circ_qubits : int, apply_qubits : List[int], n_features : int, param_prefix : str, encoder_type : str, barrier : bool) -> QuantumCircuit:
        """Generate a circuit with the specified feature encoder, prefix and number o features

        Args:
        n_features (int): Number of input features
        param_prefix (string): String prefix for encoding parameters
        encoder_type (str): Type of encoder to use "y","x", "yz", "amplitude"
        barrier (bool): Print final barrier

        Returns:
            QuantumCircuit: Encoding quantum circuit
        """
        self.functions = {
            "y" : y_angles_encoding,
            "x" : x_angles_encoding,
            "yz" : yz_angles_encoding,
            "amplitude": amplitude_encoding
        }

        m = self.functions[encoder_type](n_features, param_prefix)
        q = QuantumCircuit(circ_qubits)
        q.append(m, qargs=apply_qubits)
        if barrier:
            q.barrier()
        
        return q




class RealAmplitudeLayer(QuantumLayer):
    """RealAmplitudes layer for quantum neural networks
    """
    def __new__(
            self, 
            circ_qubits : int, 
            apply_qubits : list, 
            entanglement : str, 
            reps : int, 
            param_prefix : str, 
            layer_name : str,
            barrier : bool) -> QuantumCircuit:
        """Generate a customized RealAmplitudes ansatz

        Args:
            circ_qubits (int): Number of total qubits of the circuit
            apply_qubits (list): Qubits to which apply the RealAmplitudes ansatz
            entanglement (str): Type of entaglement (Refer to QiskitMachineLearning documentation)
            reps (int): Ansatz repetition (Refer to QiskitMachineLearning documentation)
            param_prefix (str):  String prefix for learnable ansatz parameters
            layer_name (str): Visible layer name
            barrier (bool): Print final barrier

        Returns:
            QuantumCircuit: Ansatz circuit applied on choesen qubits
        """
        
        ansatz = RealAmplitudes(len(apply_qubits), entanglement, reps, parameter_prefix=param_prefix, name=layer_name)
        qc = QuantumCircuit(circ_qubits)
        qc = qc.compose(ansatz, apply_qubits)

        if barrier:
            qc.barrier()


        return qc



class PoolingLayer(QuantumLayer):
    """Condesante information from source qubits to targent qubits using CNN inspired pooling layer
    """
    def __new__(
            self, 
            circ_qubits : int, 
            source_qubits : list, 
            target_qubits : list,
            param_prefix : str, 
            layer_name : str,
            barrier : bool) -> QuantumCircuit:
        """Generate a pooling circuit from source qubits to target qubits

        Args:
            circ_qubits (int): Number of total qubits of the circuit
            source_qubits (list): Source qubits of the pooling layer
            target_qubits (list): Target qubits of the pooling layer (the ones where information is condensed)
            param_prefix (str): Pooling rotation parameters prefix
            layer_name (str): Visible layer name
            barrier (bool): Print final barrier

        Returns:
            QuantumCircuit: Pooling operator from source qubits to target qubits
        """
        
        lanes = len(source_qubits)
        qc = QuantumCircuit(circ_qubits, name=layer_name)
        params = ParameterVector(param_prefix, length=lanes *3)

        for i in range(lanes):
            current = source_qubits[i]
            aux = target_qubits[i]
            base_param =  current*(lanes//2 -1)

            qc.rz(-np.pi/2, aux)
            qc.cx(aux, current)
            qc.rz(params[base_param + 0], current)
            qc.ry(params[base_param + 1], aux)
            qc.cx(current, aux)
            qc.ry(params[base_param + 2], aux)

        aqc = QuantumCircuit(circ_qubits)
        aqc.append(qc, range(circ_qubits))
        
        if barrier:
            aqc.barrier()

        return aqc
    

class SwapTestLayer(QuantumLayer):
    """Swap test circuit between lanes
    """
    def __new__(
            self, 
            circ_qubits : int, 
            target1_qubits : list, 
            target2_qubits : list,
            ancilla_qubit : int,
            layer_name : str,
            barrier : bool) -> QuantumCircuit:
        """Genearate swap test circuit with ancilla controll and target1 and target2 qubits

        Args:
            circ_qubits (int): Number of total qubits of the circuit
            target1_qubits (list): Swap operator first lines
            target2_qubits (list): Swap operator second lines
            ancilla_qubit (int): Control qubit for swap operation
            layer_name (str): Visible layer name
            barrier (bool): Print final barrier

        Returns:
            QuantumCircuit: Swap circuit 
        """
        
        qc = QuantumCircuit(circ_qubits, name=layer_name)

        qc.h(ancilla_qubit)
        for i in range(len(target1_qubits)):
            qc.cswap(ancilla_qubit, target1_qubits[i], target2_qubits[i])
        qc.h(ancilla_qubit)

        aqc = QuantumCircuit(circ_qubits)
        aqc.append(qc, range(circ_qubits))

        if barrier:
            aqc.barrier()

        return aqc



class QuantumSequential(QuantumLayer):
    """Sequential compositor of QuantumLayers
    """
    def __new__(self, *layers: QuantumCircuit) -> QuantumCircuit:
        """Compose multiple QuantumLayer in a single circuit

        Args:
            *layers (QuantumCircuit): Quantum laayer to be sequentially concatened

        Returns:
            QuantumCircuit: Composed quantum layer
        """
        nq = layers[0].num_qubits
        qc = QuantumCircuit(nq)
        for layer in layers:
            qc.compose(layer, range(nq), inplace=True)
        return qc

