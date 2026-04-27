"""Plot a dissociation curve from a saved CSV file."""

import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

FCI_COLOUR = '#185FA5'
ESTIMATOR_COLOUR = '#993C1D'
STYLE = 'ggplot'


def extract_shots(filename: str) -> int | None:
    """Extract the number of shots from a CSV filename, or return None."""
    match = re.search(r'_shots_(\d+)\.csv$', filename)
    return int(match.group(1)) if match else None


def plot_dissociation_curve(
    bond_lengths, vqe_energies, vqe_errors, fci_energies, name: str
):
    """Plot the dissociation curve comparing VQE results with FCI."""
    shots = extract_shots(name)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    plt.style.use(STYLE)

    plt.errorbar(
        bond_lengths, vqe_energies,
        yerr=vqe_errors,
        label='VQE (exact)' if shots is None else f'VQE (shots={shots})',
        fmt='o', capsize=4, capthick=1, elinewidth=1,
        color=ESTIMATOR_COLOUR,
    )

    plt.plot(
        bond_lengths, fci_energies,
        label='FCI (exact)', color=FCI_COLOUR,
        linestyle='--', marker='x',
    )

    plt.xlabel('Bond Length, R (Å)', fontsize=12)
    plt.ylabel('Energy (Ha)', fontsize=12)
    plt.title('Dissociation Curve - H2 (STO-3G)', fontsize=14)

    plt.grid()
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()

    out_path = RESULTS_DIR / name.replace('data', 'plot').replace('.csv', '.pdf')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a VQE dissociation curve from a CSV file.")
    parser.add_argument('--csv', type=str, required=True, help='CSV file in results/ to plot')
    args = parser.parse_args()

    csv_path = RESULTS_DIR / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"No data file found at {csv_path}")

    data = pd.read_csv(csv_path)
    plot_dissociation_curve(
        data['bond_length'], data['vqe_energy'], data['vqe_std'], data['fci_energy'],
        name=args.csv,
    )
