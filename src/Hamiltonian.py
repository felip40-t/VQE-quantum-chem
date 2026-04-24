"""
Generates the reduced qubit Hamiltonian for a molecule.

Pipeline: MolecularData -> PySCF -> FermionOperator -> Jordan-Wigner
          -> symmetry_conserving_bravyi_kitaev -> SparsePauliOp
"""

import numpy as np

from openfermion import FermionOperator, QubitOperator, count_qubits, get_fermion_operator, jordan_wigner
from openfermion.chem import MolecularData
from openfermion.transforms import symmetry_conserving_bravyi_kitaev
from openfermionpyscf import run_pyscf
from qiskit.quantum_info import SparsePauliOp


def build_molecule(geometry: list, basis="sto-3g", multiplicity=1, charge=0) -> MolecularData:
    molecule = MolecularData(geometry=geometry, basis=basis,
                             multiplicity=multiplicity, charge=charge)
    return run_pyscf(molecule, run_scf=1, run_fci=1)


def get_qubit_hamiltonian(molecule: MolecularData) -> tuple[QubitOperator, FermionOperator]:
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    return jordan_wigner(fermion_hamiltonian), fermion_hamiltonian


def get_reduced_qubit_hamiltonian(fermion_hamiltonian: FermionOperator, n_qubits: int, n_electrons: int) -> QubitOperator:
    return symmetry_conserving_bravyi_kitaev(
        fermion_hamiltonian,
        active_orbitals=n_qubits,
        active_fermions=n_electrons,
    )


def openfermion_to_qiskit(qubit_op: QubitOperator, n_qubits: int) -> SparsePauliOp:
    pauli_list = []
    for term, coeff in qubit_op.terms.items():
        if not term:
            pauli_str = "I" * n_qubits
        else:
            pauli_str = ["I"] * n_qubits
            for qubit_idx, pauli in term:
                pauli_str[qubit_idx] = pauli
            # Qiskit uses reverse qubit ordering
            pauli_str = "".join(reversed(pauli_str))
        pauli_list.append((pauli_str, coeff.real))
    return SparsePauliOp.from_list(pauli_list)


def build_reduced_hamiltonian(geometry: list, basis="sto-3g", multiplicity=1, charge=0) -> SparsePauliOp:
    """Return the symmetry-reduced qubit Hamiltonian as a Qiskit SparsePauliOp."""
    molecule = build_molecule(geometry, basis, multiplicity, charge)
    _, fermion_hamiltonian = get_qubit_hamiltonian(molecule)
    reduced_op = get_reduced_qubit_hamiltonian(
        fermion_hamiltonian, molecule.n_qubits, molecule.n_electrons
    )
    # Bravyi-Kitaev symmetry reduction drops 2 qubits from the full register
    n_reduced_qubits = count_qubits(reduced_op)
    # print(f"Original qubits: {molecule.n_qubits}, Reduced qubits: {n_reduced_qubits}")
    return openfermion_to_qiskit(reduced_op, n_reduced_qubits)


if __name__ == "__main__":
    H2_GEOMETRY = [("H", (0, 0, 0)), ("H", (0, 0, 0.735))]
    H_reduced = build_reduced_hamiltonian(H2_GEOMETRY)
    print(H_reduced)
    H_matrix = H_reduced.to_matrix()
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    print("FCI ground state energy:", eigenvalues[0])
    # Check against PySCF FCI energy
    molecule = build_molecule(H2_GEOMETRY)
    print("PySCF FCI energy:", molecule.fci_energy)
    # Nuclear repulsion energy
    print("Nuclear repulsion energy:", molecule.nuclear_repulsion)
