from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit
import math

# AMPLITUDE ENCODING
def amplitude_encoding(n_features : int, param_prefix : str) -> QuantumCircuit:
    """Amplitude encoding of n features on log2(n) qubits

    Args:
        n_features (int): Number of input features
        param_prefix (string): String prefix for encoding parameters

    Returns:
        QuantumCircuit: Encoding quantum circuit
    """
    qc = RawFeatureVector(n_features)
    qc = qc.assign_parameters(ParameterVector(param_prefix, n_features))
    qc.name = f"Amplitude Encoding {param_prefix}"
    return qc


# YZ ANGLES ENCODING
def yz_angles_encoding(n_features : int, param_prefix : str) -> QuantumCircuit:
    """YZ Angles encoding of n features on n/2 qubits

    Args:
        n_features (int): Number of input features
        param_prefix (string): String prefix for encoding parameters

    Returns:
        QuantumCircuit: Encoding quantum circuit
    """
    params = ParameterVector(param_prefix, n_features)
    n_qubit = math.floor(n_features /2) + (1 if (n_features % 2) != 0 else 0)
    qc = QuantumCircuit(n_qubit, name=f"YZ Angles Encoding {param_prefix}")
    gates = [qc.ry, qc.rz]

    for i in range(n_qubit):
        for gate_i in range(2):
            pindex = i*2 + gate_i
            if pindex < n_features:
                gates[gate_i](params[pindex], i)

    return qc


# Y ANGLES ENCODING
def y_angles_encoding(n_features : int, param_prefix : str) -> QuantumCircuit:
    """Angles encoding of n features with Y rotation on n qubits

    Args:
        n_features (int): Number of input features
        param_prefix (string): String prefix for encoding parameters

    Returns:
        QuantumCircuit: Encoding quantum circuit
    """
    params = ParameterVector(param_prefix, n_features)
    qc = QuantumCircuit(n_features, name=f"Y Angles Encoding {param_prefix}")

    for i in range(n_features):
        qc.ry(params[i], i)

    return qc


# X ANGLES ENCODING
def x_angles_encoding(n_features : int, param_prefix : str) -> QuantumCircuit:
    """Angles encoding of n features with X rotation on n qubits

    Args:
        n_features (int): Number of input features
        param_prefix (string): String prefix for encoding parameters

    Returns:
        QuantumCircuit: Encoding quantum circuit
    """
    params = ParameterVector(param_prefix, n_features)
    qc = QuantumCircuit(n_features, name=f"X Angles Encoding {param_prefix}")

    for i in range(n_features):
        qc.rx(params[i], i)

    return qc