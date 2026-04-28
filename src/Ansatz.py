""" Script to generate an ansatz class for Hardware Efficient Ansatz (HEA) and Unitary Coupled Cluster (UCC) ansatz. """

from qiskit.circuit import QuantumCircuit, ParameterVector
from abc import ABC, abstractmethod

class Ansatz(ABC):

    @abstractmethod
    def build(self) -> QuantumCircuit:
        """Build the ansatz circuit."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the ansatz."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    

class HEA(Ansatz):

    def __init__(self, n_qubits: int, n_layers: int):
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        # Each layer has 3 parameters per qubit, but there is always a final set of rotations
        # after the last layer, so we have (n_layers + 1) sets of rotations in total.
        self._num_parameters = n_qubits * (n_layers + 1) * 3  
    def build(self) -> QuantumCircuit:
        """
        Build a Hardware Efficient Ansatz (HEA) circuit.
        The HEA consists of alternating layers of single-qubit rotations and 
        entangling gates (CNOTs). Each layer applies RX, RY, and RZ rotations
        to each qubit, followed by a chain of CNOT gates.
        """
        params = ParameterVector("theta", length=self._num_parameters)
        circuit = QuantumCircuit(self._n_qubits)
        param_idx = 0
        for d in range(self._n_layers+1):
            for q in range(self._n_qubits):
                circuit.rx(params[param_idx], q)
                param_idx += 1
                circuit.ry(params[param_idx], q)
                param_idx += 1
                circuit.rz(params[param_idx], q)
                param_idx += 1
            if d < self._n_layers: # Add entangling gates between layers, but not after the last layer
                for q in range(self._n_qubits - 1):
                    circuit.cx(q, q + 1)
        return circuit

    def num_parameters(self) -> int:
        """Return the number of parameters in the HEA ansatz."""
        return self._num_parameters