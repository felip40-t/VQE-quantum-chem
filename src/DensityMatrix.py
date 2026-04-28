"""Class for a density matrix object."""

import numpy as np
import pandas as pd

from Ansatz import HEA

class DensityMatrix:
    """Class representing a density matrix."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    def from_statevector_pure(self, statevector: np.ndarray):
        """Initialize the density matrix from a statevector for a pure state."""
        self.matrix = np.outer(statevector, np.conj(statevector))

    def apply_unitary(self, unitary: np.ndarray):
        """Apply a unitary transformation to the density matrix."""
        self.matrix = unitary @ self.matrix @ np.conj(unitary.T)

    def expectation_value(self, observable: np.ndarray) -> complex:
        """Calculate the expectation value of an observable."""
        return np.trace(self.matrix @ observable)