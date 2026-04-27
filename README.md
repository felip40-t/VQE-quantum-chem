# VQE Quantum Chemistry
 
A Variational Quantum Eigensolver (VQE) implementation for computing molecular ground-state energies, with a focus on noise characterisation and density-matrix diagnostics on real IBM quantum hardware.
 
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

`src/Hamiltonian.py` implements the full pipeline from molecule geometry to a symmetry-reduced Qiskit `SparsePauliOp`. The top-level function is `build_reduced_hamiltonian`:

```python
from src.Hamiltonian import build_reduced_hamiltonian

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
python src/Hamiltonian.py
```

### Ansatz construction

`src/Ansatz.py` defines an abstract `Ansatz` base class and a concrete `HEA` (Hardware Efficient Ansatz) implementation:

```python
from src.Ansatz import HEA

ansatz = HEA(n_qubits=2, depth=1)
circuit = ansatz.build()   # returns a parameterised QuantumCircuit
print(circuit.num_parameters)  # 2 * (1+1) * 3 = 12
```

Each layer applies Rx, Ry, Rz rotations to every qubit, with CNOT entangling gates between adjacent qubits after each layer except the last.

### VQE optimisation

`src/vqe_optimiser.py` implements the full VQE loop, dissociation curve generation, and result persistence. It is driven by a CLI with four simulation modes:

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
- **`reduced_hamiltonian_from_molecule`** — a new entry point in `Hamiltonian.py` accepts a pre-built `MolecularData` object, avoiding a redundant PySCF SCF/FCI run each iteration.
- **Build AerSimulator and EstimatorV2 once only** — build the objects once before doing the bond length loop so that they aren't reconstructed for every bond length.

#### CLI reference

```bash
# Exact simulation (no shot noise)
python src/vqe_optimiser.py

# Shot noise only (500 shots)
python src/vqe_optimiser.py --shots 500

# Custom depolarising noise model
python src/vqe_optimiser.py --noise --shots 500

# IBM Vigo fake backend
python src/vqe_optimiser.py --fake-backend --shots 500
```

Results are written to `results/` as both a CSV and a PDF plot. The five dissociation curves already generated are:

- `dissociation_data_exact.csv` — exact simulation (no shot noise)
- `dissociation_data_shots_500.csv` — shot noise only, 500 shots
- `dissociation_data_noisy_shots_500.csv` — custom noise model, 500 shots
- `dissociation_data_noisy_shots_200.csv` — custom noise model, 200 shots
- `dissociation_data_fake_backend_shots_500.csv` — IBM Vigo fake backend, 500 shots

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
| 5 | Density-matrix diagnostics | ⏳ |
| 6 | Real IBM hardware runs | ⏳ |
 

---

 ## References
 - Peruzzo et al. (2013) — *A variational eigenvalue solver on a quantum processor* — arXiv:1304.3061v1
 - McArdle et al. (2020) — *Quantum computational chemistry* — arXiv:1808.10402v3 

## License
 MIT