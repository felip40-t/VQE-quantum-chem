""" Run the VQE algorithm to find the ground state energy of a molecule. """

import time
import argparse
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

from Ansatz import HEA
from Hamiltonian import build_molecule, reduced_hamiltonian_from_molecule

from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, Options
from qiskit_ibm_runtime.fake_provider import FakeVigoV2 as FakeVigo

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

NUM_POINTS = 50
DEFAULT_SHOTS = 100

def create_noise_model()->NoiseModel:
    """Create a basic noise model"""
    noise_model = NoiseModel()

    # Depolarising error for single and two-qubit gates
    error_1q = depolarizing_error(0.001, 1)
    error_2q = depolarizing_error(0.01, 2) 
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model

def vqe_objective(params: np.ndarray, circuit: QuantumCircuit, observable: SparsePauliOp, estimator: EstimatorV2) -> float:
    """Objective function for VQE optimization."""
    pub = (circuit, observable, params)
    job = estimator.run([pub])
    result = job.result()
    energy = result[0].data.evs.real
    std = result[0].data.stds.real

    return float(energy), float(std)


def run_vqe(transpiled_circuit: QuantumCircuit, observable: SparsePauliOp, estimator: EstimatorV2,
            initial_params: np.ndarray) -> tuple[float, np.ndarray]:
    """Run the VQE optimization on a pre-transpiled circuit with a layout-aligned observable."""
    result = minimize(
        lambda params, *args: vqe_objective(params, *args)[0],
        initial_params,
        args=(transpiled_circuit, observable, estimator),
        method='COBYLA',
        options={'maxiter': 100, 'rhobeg': 0.5}
    )

    return result.fun, result.x, vqe_objective(result.x, transpiled_circuit, observable, estimator)[1]

def dissociation_data(ansatz: HEA, num_points: int, noise_model: NoiseModel, shots: int, fake_backend: bool) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    """Plot the dissociation curve for H2."""
    bond_lengths = np.linspace(0.5, 3.0, num_points)
    if fake_backend:
        backend = AerSimulator.from_backend(FakeVigo(), seed_simulator=42)
    elif noise_model is None:
        backend = AerSimulator(seed_simulator=42)
    else:
        backend = AerSimulator(noise_model=noise_model, seed_simulator=42)
    
    estimator = EstimatorV2(backend, options={"default_shots": shots})

    # Pre-transpile the parameterised ansatz once; reuse for every bond length.
    raw_circuit = ansatz.build()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled_circuit = pm.run(raw_circuit)

    rng = np.random.default_rng(seed=42)
    warm_params = rng.uniform(0, 2 * np.pi, size=ansatz.num_parameters())

    vqe_energies = []
    vqe_errors = []
    fci_energies = []
    for i, r in enumerate(bond_lengths, 1):
        geometry = [("H", (0, 0, 0)), ("H", (0, 0, r))]
        molecule_data = build_molecule(geometry)
        hamiltonian = reduced_hamiltonian_from_molecule(molecule_data)
        observable = hamiltonian.apply_layout(transpiled_circuit.layout) # Align the observable with the transpiled circuit's layout, since the transpiler may have changed the qubit ordering.

        energy, warm_params, std = run_vqe(transpiled_circuit, observable, estimator, warm_params)
        vqe_energies.append(energy)
        vqe_errors.append(std)
        fci_energies.append(molecule_data.fci_energy)
        print(f"[{i}/{num_points}] r={r:.3f} Å  E={energy:.6f} Ha", flush=True)

    return bond_lengths, vqe_energies, vqe_errors, fci_energies

def make_dataframe(bond_lengths: np.ndarray, vqe_energies: list[float], vqe_errors: list[float], fci_energies: list[float]):
    """Create a pandas DataFrame from the dissociation data."""
    df = pd.DataFrame({
        'bond_length': bond_lengths,
        'vqe_energy': vqe_energies,
        'vqe_std': vqe_errors,
        'fci_energy': fci_energies
    })
    return df


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=int, default=None, help='Number of shots for the VQE estimator')
    parser.add_argument('--noise', action='store_true', help='Use a noise model for the VQE simulation')
    parser.add_argument('--fake-backend', action='store_true', help='Use a fake backend noise model based on Vigo')
    args = parser.parse_args()

    t0 = time.perf_counter()
    ansatz = HEA(n_qubits=2, depth=1)
    noise_model = create_noise_model() if args.noise else None
    bond_lengths, vqe_energies, vqe_errors, fci_energies = dissociation_data(
        ansatz,
        num_points=NUM_POINTS,
        noise_model=noise_model,
        shots=args.shots,
        fake_backend=args.fake_backend,
    )
    data = make_dataframe(bond_lengths, vqe_energies, vqe_errors, fci_energies)
    if args.noise:
        filename = f"dissociation_data_noisy_shots_{args.shots}.csv"
    elif args.fake_backend:
        filename = f"dissociation_data_fake_backend_shots_{args.shots}.csv"
    elif args.shots is None:
        filename = "dissociation_data_exact.csv"
    else:
        filename = f"dissociation_data_shots_{args.shots}.csv"

    data.to_csv(RESULTS_DIR / filename, index=False)
    elapsed = time.perf_counter() - t0
    print(f"Saved {filename}. Total runtime: {elapsed:.2f}s")
