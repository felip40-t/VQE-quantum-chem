"""Plot a dissociation curve from a saved CSV file."""

import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

STYLE = 'ggplot'

MODE_FILES = {
    'exact': 'dissociation_data_exact.csv',
    'shot':  'dissociation_data_shots_500.csv',
    'noise': 'dissociation_data_noisy_shots_500.csv',
    'fake':  'dissociation_data_fake_backend_shots_500.csv',
}

MODE_COLOURS = {
    'exact': '#2D6A4F',
    'shot':  '#993C1D',
    'noise': '#6B2D8B',
    'fake':  '#B5850B',
}
MODE_LABELS = {
    'exact': 'VQE (exact)',
    'shot':  'VQE (shot noise)',
    'noise': 'VQE (depolarising noise)',
    'fake':  'VQE (fake backend)',
}


def load_and_merge_csvs(mode_files: dict = MODE_FILES) -> pd.DataFrame:
    """Merge per-mode CSVs into a single wide DataFrame."""
    combined = None
    for mode, filename in mode_files.items():
        df = pd.read_csv(RESULTS_DIR / filename)
        df = df.rename(columns={
            'vqe_energy': f'vqe_energy_{mode}',
            'vqe_std':    f'vqe_std_{mode}',
        })
        df = df.drop(columns=['opt_params'], errors='ignore')
        if combined is None:
            combined = df
        else:
            combined = combined.merge(df.drop(columns=['fci_energy']), on='bond_length')
    return combined


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
        color='red',
    )

    plt.plot(
        bond_lengths, fci_energies,
        label='FCI (exact)', color='blue',
        linestyle='--', marker='x',
    )

    plt.xlabel('Bond Length, R (Å)', fontsize=12)
    plt.ylabel('Energy (Ha)', fontsize=12)
    plt.title('Dissociation Curve - H2 (STO-3G)', fontsize=14)

    plt.grid()
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.tight_layout()

    out_path = RESULTS_DIR / name.replace('data', 'plot').replace('.csv', '.pdf')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def plot_all_modes_curve(data: pd.DataFrame, name: str):
    """Plot all 4 VQE noise modes on the same axes alongside FCI."""
    shots = extract_shots(name)
    shots_label = f"shots={shots}" if shots else "exact"

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    plt.style.use(STYLE)

    for mode in ('exact', 'shot', 'noise', 'fake'):
        energy_col = f'vqe_energy_{mode}'
        std_col = f'vqe_std_{mode}'
        if energy_col not in data.columns:
            continue
        ax.errorbar(
            data['bond_length'], data[energy_col],
            yerr=data[std_col],
            label=MODE_LABELS[mode],
            fmt='o', capsize=4, capthick=1, elinewidth=1,
            color=MODE_COLOURS[mode],
        )

    ax.plot(
        data['bond_length'], data['fci_energy'],
        label='FCI (exact)', color="blue",
        linestyle='--', marker='x',
    )

    ax.set_xlabel('Bond Length, R (Å)', fontsize=12)
    ax.set_ylabel('Energy (Ha)', fontsize=12)
    ax.set_title(f'Dissociation Curve - H2 (STO-3G) — all modes ({shots_label})', fontsize=14)
    ax.grid()
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.tight_layout()

    out_path = RESULTS_DIR / name.replace('.csv', '_plot.pdf')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a VQE dissociation curve from a CSV file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=str, help='CSV file in results/ to plot')
    group.add_argument('--merge', action='store_true',
                       help='Merge all four mode CSVs into dissociation_data_all_modes.csv and plot')
    args = parser.parse_args()

    if args.merge:
        merged_name = 'dissociation_data_all_modes.csv'
        data = load_and_merge_csvs()
        out_path = RESULTS_DIR / merged_name
        data.to_csv(out_path, index=False)
        print(f"Saved combined CSV to {out_path}")
        plot_all_modes_curve(data, merged_name)
    else:
        csv_path = RESULTS_DIR / args.csv
        if not csv_path.exists():
            raise FileNotFoundError(f"No data file found at {csv_path}")

        data = pd.read_csv(csv_path)

        if 'vqe_energy_exact' in data.columns:
            plot_all_modes_curve(data, args.csv)
        else:
            plot_dissociation_curve(
                data['bond_length'], data['vqe_energy'], data['vqe_std'], data['fci_energy'],
                name=args.csv,
            )
