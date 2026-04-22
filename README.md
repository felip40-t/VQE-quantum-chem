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

ansatz = HEA(n_qubits=2, depth=2)
circuit = ansatz.build()   # returns a parameterised QuantumCircuit
print(circuit.num_parameters)  # 2 * (2+1) * 3 = 18
```

Each layer applies Rx, Ry, Rz rotations to every qubit, with CNOT entangling gates between adjacent qubits after each layer except the last.

### VQE optimisation

`src/vqe_optimiser.py` provides the objective function and history tracking for the VQE loop (in progress).

Both a noiseless `StatevectorEstimator` and a noisy Aer `EstimatorV2` backend are available. Energy and parameter histories are recorded each iteration for convergence analysis.

---
 
## Status
 
| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment setup | ✅ |
| 2 | Chemistry concepts & Hamiltonian generation | ✅ |
| 3 | Noiseless VQE baseline (H2) | 🔄 |
| 4 | Error decomposition with simulator | ⏳ |
| 5 | Density-matrix diagnostics | ⏳ |
| 6 | Real IBM hardware runs | ⏳ |
 

---

 ## References
 - Peruzzo et al. (2013) — *A variational eigenvalue solver on a quantum processor* — arXiv:1304.3061v1
 - McArdle et al. (2020) — *Quantum computational chemistry* — arXiv:1808.10402v3 

## License
 MIT