"""
Plot SNR vs SSIM curves for different compression ratios.

Usage:
    python plot_snr_vs_ssim.py

Output:
    ./results/figures/snr_vs_ssim.png
"""

import glob
import json
import os

import matplotlib.pyplot as plt


def load_results(results_dir="./results"):
    """Load AWGN SSIM results from evaluation JSON files."""
    results = {}
    pattern = os.path.join(results_dir, "evaluation_*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print(f"No evaluation JSON files found in {results_dir}")
        return results

    for json_file in json_files:
        filename = os.path.basename(json_file)

        try:
            ratio_str = filename.split("_kn", maxsplit=1)[1].rsplit(".json", maxsplit=1)[0]
            compression_ratio = float(ratio_str)
        except (IndexError, ValueError):
            print(f"Skipping file with unrecognized compression ratio: {filename}")
            continue

        if abs(compression_ratio - 0.0833) < 1e-8:
            print(f"Skipping {filename} because CR=0.0833 is excluded")
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable JSON file {filename}: {exc}")
            continue

        awgn_ssim = data.get("awgn", {}).get("ssim")
        if not isinstance(awgn_ssim, dict):
            print(f"Skipping file without awgn->ssim data: {filename}")
            continue

        try:
            snr_ssim = {
                float(snr): float(ssim)
                for snr, ssim in awgn_ssim.items()
            }
        except (TypeError, ValueError) as exc:
            print(f"Skipping invalid awgn->ssim data in {filename}: {exc}")
            continue

        results[compression_ratio] = dict(sorted(snr_ssim.items()))
        print(f"Loaded {filename} -> CR={compression_ratio:g}")

    return dict(sorted(results.items()))


def plot_snr_vs_ssim(results, save_path="./results/figures/snr_vs_ssim.png"):
    """Plot SNR vs SSIM curves for all loaded compression ratios."""
    if not results:
        print("No valid data available for plotting.")
        return

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for index, (compression_ratio, snr_ssim) in enumerate(results.items()):
        snr_values = list(snr_ssim.keys())
        ssim_values = list(snr_ssim.values())
        marker = markers[index % len(markers)]

        ax.plot(
            snr_values,
            ssim_values,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=f"CR={compression_ratio:.2f}",
        )

    ax.set_title("SNR vs SSIM under Different Compression Ratios")
    ax.set_xlabel("SNR (dB)")
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
    """Load results and generate the SNR vs SSIM figure."""
    results = load_results()
    plot_snr_vs_ssim(results)


if __name__ == "__main__":
    main()
