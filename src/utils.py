""" Helper functions for the main VQE algorithm. """

import numpy as np
import pandas as pd

from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)


def create_noise_model() -> NoiseModel:
    """Create a basic noise model"""
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.001, 1)
    error_2q = depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model


def vqe_objective(params: np.ndarray, circuit: QuantumCircuit, observable: SparsePauliOp, estimator: EstimatorV2) -> tuple[float, float]:
    """Objective function for VQE optimization."""
    job = estimator.run([(circuit, observable, params)])
    result = job.result()
    energy = result[0].data.evs.real
    std = result[0].data.stds.real
    return float(energy), float(std)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    filepath.parent.mkdir(exist_ok=True)
    df.to_csv(filepath, index=False)


def save_npz(bond_lengths: np.ndarray, rho_fulls: list[np.ndarray], rdm_1s: list[np.ndarray], filepath: Path) -> None:
    """Save stacked density matrices and 1-RDMs to a .npz file."""
    filepath.parent.mkdir(exist_ok=True)
    np.savez(
        filepath,
        bond_lengths=bond_lengths,
        rho_full=np.stack(rho_fulls),
        rdm_1=np.stack(rdm_1s),
    )
