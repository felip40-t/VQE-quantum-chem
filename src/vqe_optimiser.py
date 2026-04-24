""" Run the VQE algorithm to find the ground state energy of a molecule. """

import time
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

from Ansatz import HEA
from Hamiltonian import build_reduced_hamiltonian, build_molecule

from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2, Options
from qiskit_ibm_runtime.fake_provider import FakeVigoV2 as FakeVigo
from qiskit.primitives import StatevectorEstimator

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

def create_noise_model()->NoiseModel:
    """Create a basic noise model"""
    noise_model = NoiseModel()

    # Depolarising error for single and two-qubit gates
    error_1q = depolarizing_error(0.001, 1)
    error_2q = depolarizing_error(0.01, 2) 
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model

def build_fake_backend_noise_model()->(NoiseModel, FakeVigo):
    """Build a noise model based on the fake Vigo backend."""
    fake_vigo = FakeVigo()
    noise_model = NoiseModel.from_backend(fake_vigo)
    return noise_model, fake_vigo

def vqe_objective(params: np.ndarray, circuit: QuantumCircuit, hamiltonian: SparsePauliOp, estimator: EstimatorV2) -> float:
    """Objective function for VQE optimization."""
    bound_circuit = circuit.assign_parameters(params)
    pub = (bound_circuit, hamiltonian)
    job = estimator.run([pub])
    result = job.result()
    energy = result[0].data.evs.real

    return float(energy)


def run_vqe(molecule: str, ansatz: HEA, hamiltonian: SparsePauliOp, shots: int)->(float, np.ndarray):
    """Run the VQE optimization."""
    backend = AerSimulator()
    estimator = EstimatorV2(backend, options={"default_shots": shots})
    circuit = ansatz.build()

    num_params = ansatz.num_parameters()
    rng = np.random.default_rng(seed=42)
    initial_params = rng.uniform(0, 2 * np.pi, size=num_params)

    result = minimize(
        vqe_objective,
        initial_params,
        args=(circuit, hamiltonian, estimator),
        method='COBYLA',
        options={'maxiter': 250, 'rhobeg':0.5}
    )

    # print(f"VQE optimization for {molecule} completed.")
    # print(f"Optimal energy: {result.fun:.6f} Hartree")

    # Check FCI energy for comparison
    # molecule_data = build_molecule(molecule)
    # print(f"PySCF FCI energy: {molecule_data.fci_energy:.6f} Hartree")
    # print(f"Error compared to FCI: {result.fun - molecule_data.fci_energy:.6f} Hartree")

    return result.fun, result.x

def dissociation_curve(ansatz: HEA, num_points: int, noise_model: NoiseModel, shots: int):
    """Plot the dissociation curve for H2."""
    bond_lengths = np.linspace(0.5, 3.0, num_points)

    vqe_energies = []
    fci_energies = []
    for i, r in enumerate(bond_lengths, 1):
        geometry = [("H", (0, 0, 0)), ("H", (0, 0, r))]
        molecule_data = build_molecule(geometry)
        hamiltonian = build_reduced_hamiltonian(geometry)
        energy, _ = run_vqe(geometry, ansatz, hamiltonian, shots)
        vqe_energies.append(energy)
        fci_energies.append(molecule_data.fci_energy)
        print(f"[{i}/{num_points}] r={r:.3f} Å  E={energy:.6f} Ha", flush=True)

    plt.figure(figsize=(10, 6))
    plt.plot(bond_lengths, vqe_energies, label='VQE (noisy)', color='black', linestyle='-', marker='o')
    plt.plot(bond_lengths, fci_energies, label='FCI (exact)', color='red', linestyle='--', marker='x')
    plt.xlabel('Bond Length (Å)')
    plt.ylabel('Energy (Ha)')
    plt.title('Dissociation Curve - H2 (STO-3G)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'dissociation_curve_noisy.pdf')

if __name__ == "__main__":
    ansatz = HEA(n_qubits=2, depth=1)
    noise_model = create_noise_model()
    t0 = time.perf_counter()
    dissociation_curve(ansatz, num_points=20, noise_model=noise_model, shots=200)
    elapsed = time.perf_counter() - t0
    print(f"Total runtime: {elapsed:.2f}s")