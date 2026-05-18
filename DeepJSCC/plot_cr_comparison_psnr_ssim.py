"""
Compare PSNR and SSIM curves for selected compression ratios.

Usage:
    python plot_cr_comparison_psnr_ssim.py

Output:
    ./results/figures/cr_comparison_psnr_ssim.png
"""

import glob
import json
import os

import matplotlib.pyplot as plt


RESULTS_DIR = "./results"
SAVE_PATH = "./results/figures/cr_comparison_psnr_ssim.png"
TARGET_RATIOS = [0.05, 0.0833]


def _parse_compression_ratio(filename):
    """Parse compression ratio from evaluation filename."""
    ratio_str = filename.split("_kn", maxsplit=1)[1].rsplit(".json", maxsplit=1)[0]
    return float(ratio_str)


def load_results(results_dir=RESULTS_DIR):
    """Load AWGN PSNR and SSIM curves for the target compression ratios."""
    results = {}
    json_files = sorted(glob.glob(os.path.join(results_dir, "evaluation_*.json")))

    for json_file in json_files:
        filename = os.path.basename(json_file)

        try:
            compression_ratio = _parse_compression_ratio(filename)
        except (IndexError, ValueError):
            continue

        if compression_ratio not in TARGET_RATIOS:
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable JSON file {filename}: {exc}")
            continue

        awgn_data = data.get("awgn", {})
        psnr_data = awgn_data.get("psnr")
        ssim_data = awgn_data.get("ssim")
        if not isinstance(psnr_data, dict) or not isinstance(ssim_data, dict):
            print(f"Skipping file without complete awgn metric data: {filename}")
            continue

        try:
            results[compression_ratio] = {
                "psnr": dict(sorted((float(snr), float(value)) for snr, value in psnr_data.items())),
                "ssim": dict(sorted((float(snr), float(value)) for snr, value in ssim_data.items())),
            }
        except (TypeError, ValueError) as exc:
            print(f"Skipping invalid metric data in {filename}: {exc}")
            continue

        print(f"Loaded {filename} -> CR={compression_ratio:.4f}")

    return dict(sorted(results.items()))


def plot_comparison(results, save_path=SAVE_PATH):
    """Plot PSNR and SSIM comparisons for the selected compression ratios."""
    if not results:
        print("No valid data available for plotting.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    markers = ["o", "s", "^", "D"]

    for index, (compression_ratio, metrics) in enumerate(results.items()):
        marker = markers[index % len(markers)]
        label = f"CR={compression_ratio:.4f}"

        psnr_points = metrics["psnr"]
        axes[0].plot(
            list(psnr_points.keys()),
            list(psnr_points.values()),
            marker=marker,
            linewidth=2,
            markersize=6,
            label=label,
        )

        ssim_points = metrics["ssim"]
        axes[1].plot(
            list(ssim_points.keys()),
            list(ssim_points.values()),
            marker=marker,
            linewidth=2,
            markersize=6,
            label=label,
        )

    axes[0].set_title("SNR vs PSNR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("SNR vs SSIM")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def main():
    """Load results and generate the comparison figure."""
    results = load_results()
    plot_comparison(results)


if __name__ == "__main__":
    main()
