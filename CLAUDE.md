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

# Run VQE dissociation curve — exact (no noise)
python src/vqe_optimiser.py

# Run VQE dissociation curve — shot noise only
python src/vqe_optimiser.py --shots 500

# Run with custom depolarising noise model
python src/vqe_optimiser.py --noise --shots 500

# Run with IBM Vigo fake backend noise profile
python src/vqe_optimiser.py --fake-backend --shots 500

# Plot the saved csv
python src/plot_dissociation.py --csv <filename>

# Launch notebooks
jupyter lab
```

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

- `HEA(n_qubits, n_layers)` — Hardware Efficient Ansatz: Rx/Ry/Rz rotations on every qubit per layer, with CNOT ladders inserted between consecutive layers but not after the last. Parameter count: `n_qubits * (n_layers + 1) * 3` (there is always a final rotation layer after the last CNOT block). `num_parameters()` is a regular method, not a property. A single-layer circuit has no entanglement.

### 3. VQE optimisation — `src/vqe_optimiser.py`

- `vqe_objective` — wraps a parameterised circuit + Hamiltonian into a scalar energy and std via `EstimatorV2.run`.
- `run_vqe` — calls `scipy.optimize.minimize` with COBYLA (100 iterations, `rhobeg=0.5`, seed 42).
- `dissociation_data` — sweeps bond length 0.5–3.0 Å over 50 points; transpiles the ansatz once before the loop and warm-starts each optimisation from the previous bond length's optimal parameters.
- `make_dataframe` — builds a pandas DataFrame from dissociation data (columns: `bond_length`, `vqe_energy`, `vqe_std`, `fci_energy`, `opt_params`); saved as CSV before the process exits.
- CLI entry point supports four simulation modes via `--noise`, `--fake-backend`, and `--shots` flags.

#### Performance optimisations
- **Single transpilation**: the parameterised ansatz is transpiled once with `generate_preset_pass_manager` and reused for every geometry, eliminating per-step compilation overhead.
- **Warm-start parameters**: optimal parameters from step `i` seed step `i+1`, reducing COBYLA iterations needed per bond length.
- **`reduced_hamiltonian_from_molecule`** in `Hamiltonian.py`: accepts a pre-built `MolecularData` to avoid re-running PySCF inside the sweep loop.
- **Single backend/estimator construction**: `AerSimulator` and `EstimatorV2` are built once before the bond-length loop, not reconstructed per step.

#### Noise backends
| Flags | Backend |
|-------|---------|
| *(no flags, `--shots` omitted)* | `AerSimulator` — exact (shots=None) |
| `--shots N` | `AerSimulator` — shot noise only |
| `--noise --shots N` | `AerSimulator` + `create_noise_model()` — depolarising (0.1 % 1Q, 1 % 2Q) |
| `--fake-backend --shots N` | `AerSimulator.from_backend(FakeVigoV2)` — IBM Vigo device noise profile |

### 4. Density-matrix diagnostics — `src/DensityMatrix.py`

`DensityMatrix(n_qubits)` — stores a complex `(2^n, 2^n)` density matrix and exposes:

- `from_statevector_pure(psi)` — initialises `rho = |psi><psi|` for pure states.
- `apply_unitary(U)` — updates `rho -> U rho U†`.
- `expectation_value(O)` — returns `Tr(rho O)`.

### 5. Dissociation curve plotting — `src/plot_dissociation.py`

Standalone plotting utility; does not import any local VQE modules, so no `PYTHONPATH` is needed.

- `extract_shots(filename)` — parses shot count from a CSV filename; returns `None` for exact runs.
- `plot_dissociation_curve(bond_lengths, vqe_energies, vqe_errors, fci_energies, name)` — overlays VQE energies (with error bars) against the FCI reference and saves a PDF to `results/`.
- CLI: `python src/plot_dissociation.py --csv <filename>` — regenerates a plot from any saved CSV without re-running the simulation.

### Qubit counting convention

OpenFermion and Qiskit use opposite qubit-index orderings. The conversion in `openfermion_to_qiskit` reverses the Pauli string to account for this.
