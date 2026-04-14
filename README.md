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
 
> ⚠️ Implementation in progress. This section will be updated as phases are completed.
 
---
 
## Status
 
| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment setup | ✅ |
| 2 | Chemistry concepts & Hamiltonian generation | 🔄 In progress |
| 3 | Noiseless VQE baseline (H2) | ⏳ |
| 4 | Error decomposition with simulator | ⏳ |
| 5 | Density-matrix diagnostics | ⏳ |
| 6 | Real IBM hardware runs | ⏳ |
 
---
 ## References
 - Peruzzo et al. (2013) — *A variational eigenvalue solver on a quantum processor* — arXiv:1304.3061v1
 - McArdle et al. (2020) — *Quantum computational chemistry* — arXiv:1808.10402v3 

## License
 MIT