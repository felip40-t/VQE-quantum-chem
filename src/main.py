""" Run the VQE algorithm to find the ground state energy of a molecule. """

import time
import argparse
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from pathlib import Path

from Ansatz import HEA
from hamiltonian import build_molecule, reduced_hamiltonian_from_molecule
from utils import create_noise_model, vqe_objective, save_csv, save_npz
from reconstruct import measure_exp_vals, reconstruct_density_matrix, reconstruct_1rdm

from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeVigoV2 as FakeVigo
from qiskit_aer import AerSimulator

RESULTS_DIR = Path(__file__).parent.parent / "results" / "h2" / "data"
NUM_POINTS = 50

def run_vqe(transpiled_circuit: QuantumCircuit, observable: SparsePauliOp, estimator: EstimatorV2,
            initial_params: np.ndarray) -> tuple[float, np.ndarray]:
    """Run the VQE optimization on a pre-transpiled circuit with a layout-aligned observable."""
    last_std = [0.0]

    def objective(params):
        energy, std = vqe_objective(params, transpiled_circuit, observable, estimator)
        last_std[0] = std
        return energy

    result = minimize(objective, initial_params, method='COBYLA', options={'maxiter': 100, 'rhobeg': 0.5})
    return result.fun, result.x, last_std[0]

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
    warm_params = rng.uniform(0, 2 * np.pi, size=ansatz.num_parameters)

    vqe_energies = []
    vqe_errors = []
    fci_energies = []
    opt_params = []
    rho_fulls = []
    rdm_1s = []
    for i, r in enumerate(bond_lengths, 1):
        geometry = [("H", (0, 0, 0)), ("H", (0, 0, r))]
        molecule_data = build_molecule(geometry)
        hamiltonian = reduced_hamiltonian_from_molecule(molecule_data)
        # Align the observable with the transpiled circuit's layout, since the transpiler may have changed the qubit ordering.
        observable = hamiltonian.apply_layout(transpiled_circuit.layout) 

        energy, warm_params, std = run_vqe(transpiled_circuit, observable, estimator, warm_params)

        bound_circuit = transpiled_circuit.assign_parameters(warm_params)
        exp_vals_reconst, stds_reconst = measure_exp_vals(estimator, bound_circuit, ansatz.n_qubits)
        rho_full = reconstruct_density_matrix(exp_vals_reconst, ansatz.n_qubits)
        rdm_1 = reconstruct_1rdm(rho_full, ansatz.n_qubits, molecule_data.n_electrons)

        vqe_energies.append(energy)
        vqe_errors.append(std)
        opt_params.append(warm_params)
        fci_energies.append(molecule_data.fci_energy)
        rho_fulls.append(rho_full.matrix)
        rdm_1s.append(rdm_1)

        print(f"[{i}/{num_points}] r={r:.3f} Å  E={energy:.6f} Ha", flush=True)

    return bond_lengths, vqe_energies, vqe_errors, fci_energies, opt_params, rho_fulls, rdm_1s

def run_all_modes(ansatz: HEA, num_points: int, shots: int) -> None:
    """Run all 4 noise modes and save per-mode CSV and NPZ files."""
    modes = {
        'exact': {'noise_model': None, 'shots': None,  'fake_backend': False},
        'shot':  {'noise_model': None, 'shots': shots, 'fake_backend': False},
        'noise': {'noise_model': create_noise_model(), 'shots': shots, 'fake_backend': False},
        'fake':  {'noise_model': None, 'shots': shots, 'fake_backend': True},
    }

    fci_saved = False
    for mode_name, kwargs in modes.items():
        print(f"\n=== Mode: {mode_name} ===", flush=True)
        bond_lengths, vqe_energies, vqe_errors, fci_energies, opt_params, rho_fulls, rdm_1s = dissociation_data(
            ansatz, num_points, **kwargs
        )
        shots_str = f"_shots_{shots}" if kwargs['shots'] else ""
        stem = f"{mode_name}{shots_str}"

        if not fci_saved:
            fci_df = pd.DataFrame({'bond_length': bond_lengths, 'fci_energy': fci_energies})
            save_csv(fci_df, RESULTS_DIR / "fci_energies.csv")
            fci_saved = True

        vqe_df = pd.DataFrame({
            'bond_length': bond_lengths,
            'vqe_energy': vqe_energies,
            'vqe_std': vqe_errors,
            'opt_params': [p.tolist() for p in opt_params],
        })
        save_csv(vqe_df, RESULTS_DIR / f"dissociation_data_{stem}.csv")
        save_npz(bond_lengths, rho_fulls, rdm_1s, RESULTS_DIR / f"density_matrices_{stem}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=int, default=None, help='Number of shots for the VQE estimator')
    parser.add_argument('--noise', action='store_true', help='Apply depolarising noise model')
    parser.add_argument('--fake-backend', action='store_true', help='Use IBM Vigo fake backend noise profile')
    parser.add_argument('--all', action='store_true', help='Run all 4 noise modes and save a combined CSV')
    args = parser.parse_args()

    t0 = time.perf_counter()
    ansatz = HEA(n_qubits=2, n_layers=1)

    if args.all:
        if args.shots is None:
            print("Error: --all requires --shots to specify finite shots for the noisy modes.")
            exit(1)
        run_all_modes(ansatz, NUM_POINTS, args.shots)
    else:
        if args.shots is None and (args.noise or args.fake_backend):
            print("Warning: Noise models have no effect without --shots.")
            exit(1)
        noise_model = create_noise_model() if args.noise else None
        bond_lengths, vqe_energies, vqe_errors, fci_energies, opt_params, rho_fulls, rdm_1s = dissociation_data(
            ansatz, NUM_POINTS, noise_model, args.shots, args.fake_backend
        )
        if args.shots is None:
            mode_tag = 'exact'
        elif args.fake_backend:
            mode_tag = 'fake'
        elif args.noise:
            mode_tag = 'noise'
        else:
            mode_tag = 'shot'
        shots_str = f"_shots_{args.shots}" if args.shots else ""
        stem = f"{mode_tag}{shots_str}"

        fci_df = pd.DataFrame({'bond_length': bond_lengths, 'fci_energy': fci_energies})
        save_csv(fci_df, RESULTS_DIR / "fci_energies.csv")

        vqe_df = pd.DataFrame({
            'bond_length': bond_lengths,
            'vqe_energy': vqe_energies,
            'vqe_std': vqe_errors,
            'opt_params': [p.tolist() for p in opt_params],
        })
        save_csv(vqe_df, RESULTS_DIR / f"dissociation_data_{stem}.csv")
        save_npz(bond_lengths, rho_fulls, rdm_1s, RESULTS_DIR / f"density_matrices_{stem}.npz")

    elapsed = time.perf_counter() - t0
    print(f"Total runtime: {elapsed:.2f}s")
