"""
Plot test SNR vs SSIM under different training SNR settings.

Usage:
    python plot_train_snr_vs_ssim.py

Output:
    ./results/figures/train_snr_vs_ssim.png
"""

import os

import matplotlib.pyplot as plt

from train_snr_experiment_utils import load_results


SAVE_PATH = "./results/figures/train_snr_vs_ssim.png"


def plot_ssim(results, save_path=SAVE_PATH):
    """Plot SSIM curves for different training SNR values."""
    if not results:
        print("No valid SSIM data available for plotting.")
        return

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for index, (train_snr, snr_ssim) in enumerate(results.items()):
        test_snrs = list(snr_ssim.keys())
        ssim_values = list(snr_ssim.values())
        ax.plot(
            test_snrs,
            ssim_values,
            marker=markers[index % len(markers)],
            linewidth=2,
            markersize=6,
            label=f"Train SNR={train_snr}dB",
        )

    ax.set_title("Test SNR vs SSIM under Different Training SNR")
    ax.set_xlabel("Test SNR (dB)")
    ax.set_ylabel("SSIM")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def main():
    """Load SSIM results and generate the figure."""
    results = load_results(metric="ssim")
    plot_ssim(results)


if __name__ == "__main__":
    main()
