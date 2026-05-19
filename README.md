# VQE Quantum Chemistry
 
A Variational Quantum Eigensolver (VQE) implementation for computing molecular ground-state energies, with a focus on noise characterisation and density-matrix diagnostics on real IBM quantum hardware.

Version: 2.2.0
 
---
 
## Aims & Objectives
 
- Implement a noiseless VQE baseline for H2 (STO-3G) and benchmark against classical FCI/CCSD results
- Decompose hardware vs. simulator error sources to understand the practical limits of near-term quantum devices
- Apply density-matrix diagnostics to analyse the physicality of reconstructed states from noisy measurements
- Validate results on real IBM quantum hardware via Qiskit IBM Runtime
 
---

## Technologies
 
| Tool | Purpose |
|------|---------|
| [PySCF](https://pyscf.org/) | Molecular Hamiltonian generation (STO-3G basis) |
| [OpenFermion](https://github.com/quantumlib/OpenFermion) | Fermionic and qubit operator algebra |
| [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF) | PySCF integration for electronic structure |
| [Qiskit](https://qiskit.org/) | Quantum circuit construction and simulation |
| [qiskit-ibm-runtime](https://github.com/Qiskit/qiskit-ibm-runtime) | IBM hardware execution and noise models |
| [NumPy / SciPy](https://scipy.org/) | Numerical routines, optimisers (COBYLA, L-BFGS-B) |
| [pandas](https://pandas.pydata.org/) | Results persistence (CSV I/O, DataFrame construction) |
| [Matplotlib](https://matplotlib.org/) | Results visualisation |
 
---
## Installation
 
### Prerequisites
 
- Python 3.10+
- An [IBM Quantum account](https://quantum.ibm.com/)
 
### Setup
 
```bash
git clone https://github.com/felip40-t/vqe-quantum-chemistry.git
cd vqe-quantum-chemistry
 
python -m venv .venv
source .venv/bin/activate        # Windows: venv\Scripts\activate
 
pip install -r requirements.txt
```
### IBM Quantum credentials
 
```bash
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_API_TOKEN')
"
```
 
---
 
## Usage

### Generating the reduced qubit Hamiltonian

`src/hamiltonian.py` implements the full pipeline from molecule geometry to a symmetry-reduced Qiskit `SparsePauliOp`. The top-level function is `build_reduced_hamiltonian`:

```python
from src.hamiltonian import build_reduced_hamiltonian

H2_GEOMETRY = [("H", (0, 0, 0)), ("H", (0, 0, 0.735))]
H = build_reduced_hamiltonian(H2_GEOMETRY)
```

The pipeline runs in four stages:

1. **`build_molecule`** — constructs a `MolecularData` object and runs PySCF (SCF + FCI) to obtain one- and two-electron integrals.
2. **`get_qubit_hamiltonian`** — converts the molecular Hamiltonian to a `FermionOperator` via `get_fermion_operator`, then applies the Jordan-Wigner transformation to produce a 4-qubit `QubitOperator`.
3. **`get_reduced_qubit_hamiltonian`** — applies `symmetry_conserving_bravyi_kitaev` to exploit particle-number and spin symmetries, reducing the register from 4 qubits to 2 qubits (5 Pauli terms).
4. **`openfermion_to_qiskit`** — converts the `QubitOperator` to a Qiskit `SparsePauliOp`, handling the index-reversal convention between the two libraries.

The script can also be run directly as a sanity check:

```bash
python src/hamiltonian.py
```

### Ansatz construction

`src/Ansatz.py` defines an abstract `Ansatz` base class and a concrete `HEA` (Hardware Efficient Ansatz) implementation:

```python
from src.Ansatz import HEA

ansatz = HEA(n_qubits=2, n_layers=1)
circuit = ansatz.build()   # returns a parameterised QuantumCircuit
print(ansatz.num_parameters())  # 2 * 2 * 3 = 12
```

Each layer applies Rx, Ry, Rz rotations to every qubit. CNOT entangling gates are inserted between consecutive layers but not after the final layer, so a single-layer ansatz (`n_layers=1`) has no entanglement.

### VQE optimisation

`src/main.py` implements the full VQE loop, dissociation curve generation, density-matrix reconstruction, and result persistence. It is driven by a CLI with four simulation modes:

| Mode | Flags | Backend |
|------|-------|---------|
| Exact | *(no noise flags, `--shots` omitted)* | `AerSimulator` (noiseless, infinite precision, shots=None) |
| Shot noise only | `--shots N` | `AerSimulator` (finite shots, no gate errors) |
| Custom noise model | `--noise --shots N` | `AerSimulator` + depolarising errors (0.1 % 1Q, 1 % 2Q) |
| Fake backend | `--fake-backend --shots N` | `AerSimulator.from_backend(FakeVigoV2)` — IBM Vigo noise profile |

#### Key performance optimisations

- **Single transpilation** — the parameterised ansatz circuit is transpiled once before the bond-length sweep and reused at every point, avoiding repeated compilation overhead.
- **Warm-start parameters** — the optimal parameters from one bond length are passed as the starting point for the next, which cuts the number of COBYLA iterations needed to converge.
- **Reduced iteration budget** — `maxiter` lowered from 500 to 100 with `rhobeg=0.5`; convergence is reliable given the warm start.
- **`reduced_hamiltonian_from_molecule`** — entry point in `hamiltonian.py` that accepts a pre-built `MolecularData` object, avoiding a redundant PySCF SCF/FCI run each iteration.
- **Build AerSimulator and EstimatorV2 once only** — build the objects once before doing the bond length loop so that they aren't reconstructed for every bond length.

#### CLI reference

```bash
# Exact simulation (no shot noise)
python src/main.py

# Shot noise only (500 shots)
python src/main.py --shots 500

# Custom depolarising noise model
python src/main.py --noise --shots 500

# IBM Vigo fake backend
python src/main.py --fake-backend --shots 500

# Run all four modes in one go
python src/main.py --all --shots 500
```

Results are written to `results/` as:
- **CSV** per mode — columns: `bond_length`, `vqe_energy`, `vqe_std`, `opt_params`; FCI reference saved separately as `fci_energies.csv`
- **NPZ** per mode — stacked arrays: `bond_lengths`, `rho_full` (full density matrix at each bond length), `rdm_1` (1-RDM)

### Density-matrix diagnostics

Two modules handle density-matrix work:

**`src/DensityMatrix.py`** — lightweight container class:

```python
from src.DensityMatrix import DensityMatrix

dm = DensityMatrix(n_qubits=2)
dm.from_statevector_pure(statevector)      # initialise from a pure statevector
dm.apply_unitary(unitary)                  # propagate under a unitary
energy = dm.expectation_value(observable)  # Tr(rho * O)
```

| Method | Description |
|--------|-------------|
| `from_statevector_pure(psi)` | Sets `rho = |psi><psi|` |
| `apply_unitary(U)` | Updates `rho -> U rho U†` |
| `expectation_value(O)` | Returns `Tr(rho O)` |

**`src/reconstruct.py`** — Pauli tomography and 1-RDM reconstruction. During the VQE sweep, `main.py` calls these to reconstruct the full state and reduced density matrix at each bond length:

- `measure_exp_vals` — measures expectation values of all `4^n` Pauli operators via `EstimatorV2`
- `reconstruct_density_matrix` — expands `rho` in the Pauli basis from those expectation values
- `reconstruct_1rdm` — contracts the full density matrix to the 1-body reduced density matrix using the Bravyi-Kitaev mapped `a_p† a_q` operators

### Plotting dissociation curves

`src/plot_dissociation.py` is a standalone plotting utility that regenerates a PDF dissociation curve from any saved CSV without re-running the simulation. It reads from `results/` and writes the output PDF back to `results/`, inferring the shot count from the filename automatically.

```python
from src.plot_dissociation import plot_dissociation_curve

plot_dissociation_curve(
    bond_lengths, vqe_energies, vqe_errors, fci_energies,
    name="dissociation_data_shots_500.csv",
)
```

The script can also be invoked directly from the CLI:

```bash
# Re-plot from any saved CSV
python src/plot_dissociation.py --csv dissociation_data_shots_500.csv
```

Each plot overlays VQE energies (with shot-noise error bars) against the exact FCI reference curve.

---
 
## Status
 
| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment setup | ✅ |
| 2 | Chemistry concepts & Hamiltonian generation | ✅ |
| 3 | VQE baseline and dissociation curves (H2, STO-3G) | ✅ |
| 4 | Error decomposition with simulator | 🔄 |
| 5 | Density-matrix diagnostics | 🔄 |
| 6 | Real IBM hardware runs | ⏳ |
 

---

 ## References
 - Peruzzo et al. (2013) — *A variational eigenvalue solver on a quantum processor* — arXiv:1304.3061v1
 - McArdle et al. (2020) — *Quantum computational chemistry* — arXiv:1808.10402v3 

## License
 MIT