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

    def __init__(self, n_qubits: int, depth: int):
        self._n_qubits = n_qubits
        self.depth = depth
        self._num_parameters = n_qubits * (depth + 1) * 3  # Each layer has 3 parameters per qubit

    def build(self) -> QuantumCircuit:
        """Build a Hardware Efficient Ansatz (HEA) circuit."""
        params = ParameterVector("theta", length=self._num_parameters)
        circuit = QuantumCircuit(self._n_qubits)
        param_idx = 0
        for d in range(self.depth + 1):
            for q in range(self._n_qubits):
                circuit.rx(params[param_idx], q)
                param_idx += 1
                circuit.ry(params[param_idx], q)
                param_idx += 1
                circuit.rz(params[param_idx], q)
                param_idx += 1
            if d < self.depth:
                for q in range(self._n_qubits - 1):
                    circuit.cx(q, q + 1)
        return circuit

    def num_parameters(self) -> int:
        """Return the number of parameters in the HEA ansatz."""
        return self._num_parameters