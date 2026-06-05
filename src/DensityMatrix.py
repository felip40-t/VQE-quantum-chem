"""Class for a density matrix object."""

import numpy as np

class DensityMatrix:
    """Class representing a density matrix."""
    def __init__(self, n_qubits: int):
        self._n_qubits = n_qubits
        self._matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    @property
    def n_qubits(self) -> int:
        """Return the number of qubits."""
        return self._n_qubits

    @property
    def matrix(self) -> np.ndarray:
        """Return the density matrix."""
        return self._matrix
    
    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        dim = 2 ** self._n_qubits
        if value.shape != (dim, dim):
            raise ValueError(f"Input matrix must be of shape ({dim}, {dim}) for n_qubits={self._n_qubits}.")
        self._matrix = value

    def from_statevector_pure(self, statevector: np.ndarray) -> None:
        """Initialize the density matrix from a statevector for a pure state."""
        # Check dimension of the statevector
        if statevector.shape != (2**self._n_qubits,):
            raise ValueError(f"Statevector must be of shape ({2**self._n_qubits},) for n_qubits={self._n_qubits}.")
        self.matrix = np.outer(statevector, np.conj(statevector))

    def apply_unitary(self, unitary: np.ndarray) -> None:
        """Apply a unitary transformation to the density matrix."""
        # Check dimension of the unitary
        if unitary.shape != (2**self._n_qubits, 2**self._n_qubits):
            raise ValueError(f"Unitary must be of shape ({2**self._n_qubits}, {2**self._n_qubits}) for n_qubits={self._n_qubits}.")
        self.matrix = unitary @ self._matrix @ np.conj(unitary.T)

    def expectation_value(self, observable: np.ndarray) -> complex:
        """Calculate the expectation value of an observable."""
        # Check dimension of the observable
        if observable.shape != (2**self._n_qubits, 2**self._n_qubits):
            raise ValueError(f"Observable must be of shape ({2**self._n_qubits}, {2**self._n_qubits}) for n_qubits={self._n_qubits}.")
        return np.trace(self._matrix @ observable)

    def entropy(self) -> float:
        """Calculate the von Neumann entropy of the density matrix."""
        eigenvalues = np.linalg.eigvalsh(self._matrix)
        # Filter out zero eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return -np.sum(eigenvalues * np.log(eigenvalues))

    def purity(self) -> float:
        """Calculate the purity of the density matrix."""
        return np.trace(self._matrix @ self._matrix).real

    def check_validity(self) -> bool:
        """Check if the density matrix is valid (positive semidefinite, hermitian and trace 1)."""
        # Check if the matrix is Hermitian
        if not np.allclose(self._matrix, self._matrix.conj().T):
            print("DensityMatrix Error: Density matrix is not Hermitian.")
            return False
        # Check if the matrix is positive semidefinite
        if np.any(np.linalg.eigvalsh(self._matrix) < -1e-12):
            print("DensityMatrix Error: Density matrix is not positive semidefinite.")
            return False
        # Check if the trace is 1
        if not np.isclose(np.trace(self._matrix), 1):
            print("DensityMatrix Error: Density matrix is not normalised.")
            return False
        return True

    def frobenius_distance(self, other: 'DensityMatrix') -> float:
        """Calculate the Frobenius distance between this density matrix and another."""
        return np.linalg.norm(self._matrix - other._matrix, 'fro')

    def higham_project(self) -> 'DensityMatrix':
        """Return the nearest valid density matrix using Higham's projection."""
        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self._matrix)
        # Clip negative eigenvalues to zero
        eigenvalues = np.maximum(eigenvalues, 0)
        # Reconstruct the matrix
        projected_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        # Normalize to ensure trace 1
        projected_matrix /= np.trace(projected_matrix)
        # Create a new density matrix with the projected matrix
        new_matrix = DensityMatrix(self._n_qubits)
        new_matrix.matrix = projected_matrix
        return new_matrix
