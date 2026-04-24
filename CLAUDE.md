# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Constraints

- Do NOT delete any files, ever. If removal is needed, ask the user to do it manually.
- Do NOT read or edit files outside the project root directory. All file operations must stay within the project folder.


## Commands

```bash
# Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run Hamiltonian pipeline sanity check
python src/Hamiltonian.py

# Run VQE optimisation (runs dissociation curve after single-point)
python src/vqe_optimiser.py

# Launch notebooks
jupyter lab
```

`src/vqe_optimiser.py` imports `Ansatz` and `Hamiltonian` with bare module names (no `src.` prefix), so it must be run from the `src/` directory or with `PYTHONPATH=src`.

## Architecture

The codebase implements a VQE pipeline in three stages, each encapsulated in its own module:

### 1. Hamiltonian generation — `src/Hamiltonian.py`

Converts a molecule geometry to a symmetry-reduced 2-qubit `SparsePauliOp` via:

- PySCF (SCF + FCI) → `MolecularData`
- Jordan-Wigner → 4-qubit `QubitOperator`
- `symmetry_conserving_bravyi_kitaev` → 2-qubit `QubitOperator`
- Index reversal (OpenFermion ↔ Qiskit qubit ordering) → `SparsePauliOp`

The top-level entry point is `build_reduced_hamiltonian(geometry)`.

### 2. Ansatz construction — `src/Ansatz.py`

Abstract base class `Ansatz` with one concrete implementation:

- `HEA(n_qubits, depth)` — Hardware Efficient Ansatz: Rx/Ry/Rz rotations on every qubit per layer, with CNOT ladders between layers. Parameter count: `n_qubits * (depth + 1) * 3`.

### 3. VQE optimisation — `src/vqe_optimiser.py`

- `vqe_objective` — wraps a parameterised circuit + Hamiltonian into a scalar energy via Qiskit `Estimator.run`.
- `run_vqe` — calls `scipy.optimize.minimize` with COBYLA (500 iterations, seed 42).
- `dissociation_curve` — sweeps bond length 0.5–3.0 Å, runs VQE at each point, saves a PDF to `results/`.

Two estimators are available at module level: `exact_estimator` (`StatevectorEstimator`) and `noisy_estimator` (`EstimatorV2` from `FakeVigoV2`). The default for all functions is `noisy_estimator`.

### Qubit counting convention

OpenFermion and Qiskit use opposite qubit-index orderings. The conversion in `openfermion_to_qiskit` reverses the Pauli string to account for this.
