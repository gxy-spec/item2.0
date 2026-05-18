"""
Plot test SNR vs PSNR under different training SNR settings.

Usage:
    python plot_train_snr_vs_psnr.py

Output:
    ./results/figures/train_snr_vs_psnr.png
"""

import os

import matplotlib.pyplot as plt

from train_snr_experiment_utils import load_results


SAVE_PATH = "./results/figures/train_snr_vs_psnr.png"


def plot_psnr(results, save_path=SAVE_PATH):
    """Plot PSNR curves for different training SNR values."""
    if not results:
        print("No valid PSNR data available for plotting.")
        return

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for index, (train_snr, snr_psnr) in enumerate(results.items()):
        test_snrs = list(snr_psnr.keys())
        psnr_values = list(snr_psnr.values())
        ax.plot(
            test_snrs,
            psnr_values,
            marker=markers[index % len(markers)],
            linewidth=2,
            markersize=6,
            label=f"Train SNR={train_snr}dB",
        )

    ax.set_title("Test SNR vs PSNR under Different Training SNR")
    ax.set_xlabel("Test SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def main():
    """Load PSNR results and generate the figure."""
    results = load_results(metric="psnr")
    plot_psnr(results)


if __name__ == "__main__":
    main()
