""" Script to reconstruct the density matrix from measurement results of pauli operators. """

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2
from DensityMatrix import DensityMatrix
from hamiltonian import map_fermionic_op_to_qubit_op, qubit_op_to_matrix

LABELS = ['I', 'X', 'Y', 'Z']

def construct_labels(num_qubits: int) -> list[str]:
    """Construct the labels for the Pauli operators."""
    if num_qubits == 1:
        return LABELS
    else:
        return [a + b for a in construct_labels(num_qubits - 1) for b in LABELS]
    

def measure_exp_vals(estimator: EstimatorV2, circuit: QuantumCircuit, num_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Measure the expectation values of all Pauli operators."""
    labels = construct_labels(num_qubits)
    pauli_ops = [SparsePauliOp(label) for label in labels]
    # The circuit may have been transpiled onto a wider device (e.g. 5-qubit fake backend);
    # align the observables to its layout so qubit counts match, as in the VQE path.
    if circuit.layout is not None:
        pauli_ops = [op.apply_layout(circuit.layout) for op in pauli_ops]
    job = estimator.run([(circuit, pauli_ops)])
    result = job.result()[0].data.evs
    std = job.result()[0].data.stds
    return np.array(result), np.array(std)

def reconstruct_density_matrix(exp_vals: np.ndarray, num_qubits: int) -> DensityMatrix:
    """Reconstruct the density matrix from the expectation values."""
    dim = 2 ** num_qubits
    labels = construct_labels(num_qubits)
    pauli_ops = [SparsePauliOp(label) for label in labels]
    rho = np.zeros((dim, dim), dtype=complex)
    for exp_val, pauli in zip(exp_vals, pauli_ops):
        rho += exp_val * pauli.to_matrix()
    dm = DensityMatrix(num_qubits)
    dm.matrix = rho / dim
    return dm

def build_1rdm_operators(num_qubits: int, n_electrons: int) -> list[SparsePauliOp]:
    """Build the list of 1-RDM operators in the qubit basis."""
    operators = {}
    for p in range(num_qubits):
        for q in range(num_qubits):
            op = map_fermionic_op_to_qubit_op(p, q, num_qubits, n_electrons)
            operators[(p, q)] = op
    return operators

def reconstruct_1rdm(rho: DensityMatrix, num_qubits: int, n_electrons: int) -> np.ndarray:
    """Reconstruct the 1-RDM from the full density matrix."""
    rdm = np.zeros((num_qubits, num_qubits), dtype=complex)
    operators = build_1rdm_operators(num_qubits, n_electrons)
    for (p, q), op in operators.items():
        op_matrix = qubit_op_to_matrix(op, num_qubits)
        rdm[p, q] = np.trace(rho.matrix @ op_matrix)
    return rdm

def frobenius_distance(rdm1: np.ndarray, rdm2: np.ndarray) -> float:
    """Calculate the Frobenius distance between two 1-RDMs."""
    return np.linalg.norm(rdm1 - rdm2, 'fro')

def higham_project(rdm: np.ndarray) -> np.ndarray:
    """Project a 1-RDM onto the nearest valid 1-RDM using Higham's projection."""
    # eigen-decompose the matrix
    eigenvalues, eigenvectors = np.linalg.eigh(rdm)
    # set negative eigenvalues to zero
    eigenvalues = np.maximum(eigenvalues, 0)
    # reconstruct the matrix
    projected_rdm = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    # normalize to ensure trace equals number of electrons
    projected_rdm *= rdm.trace() / projected_rdm.trace()
    return projected_rdm
