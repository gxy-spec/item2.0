"""
Plot SNR vs PSNR curves for different compression ratios.

Usage:
    python plot_snr_vs_psnr.py

Output:
    ./results/figures/snr_vs_psnr.png
"""

import glob
import json
import os

import matplotlib.pyplot as plt


def load_results(results_dir="./results"):
    """Load AWGN PSNR results from evaluation JSON files.

    Args:
        results_dir (str): Directory that stores evaluation JSON files.

    Returns:
        dict[float, dict[float, float]]: Mapping from compression ratio to
            {snr_db: psnr_db}.
    """
    results = {}
    pattern = os.path.join(results_dir, "evaluation_*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print(f"No evaluation JSON files found in {results_dir}")
        return results

    for json_file in json_files:
        filename = os.path.basename(json_file)

        try:
            # Example:
            # evaluation_deep_jscc_cifar10_snr10_awgn_kn0.0500.json -> 0.05
            ratio_str = filename.split("_kn", maxsplit=1)[1].rsplit(".json", maxsplit=1)[0]
            compression_ratio = float(ratio_str)
        except (IndexError, ValueError):
            print(f"Skipping file with unrecognized compression ratio: {filename}")
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable JSON file {filename}: {exc}")
            continue

        awgn_psnr = data.get("awgn", {}).get("psnr")
        if not isinstance(awgn_psnr, dict):
            print(f"Skipping file without awgn->psnr data: {filename}")
            continue

        try:
            snr_psnr = {
                float(snr): float(psnr)
                for snr, psnr in awgn_psnr.items()
            }
        except (TypeError, ValueError) as exc:
            print(f"Skipping invalid awgn->psnr data in {filename}: {exc}")
            continue

        results[compression_ratio] = dict(sorted(snr_psnr.items()))
        print(f"Loaded {filename} -> CR={compression_ratio:g}")

    return dict(sorted(results.items()))


def plot_snr_vs_psnr(results, save_path="./results/figures/snr_vs_psnr.png"):
    """Plot SNR vs PSNR curves for all loaded compression ratios."""
    if not results:
        print("No valid data available for plotting.")
        return

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for index, (compression_ratio, snr_psnr) in enumerate(results.items()):
        snr_values = list(snr_psnr.keys())
        psnr_values = list(snr_psnr.values())
        marker = markers[index % len(markers)]

        ax.plot(
            snr_values,
            psnr_values,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=f"CR={compression_ratio:.2f}",
        )

    ax.set_title("SNR vs PSNR under Different Compression Ratios")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to {save_path}")


def main():
    """Load results and generate the SNR vs PSNR figure."""
    results = load_results()
    plot_snr_vs_psnr(results)


if __name__ == "__main__":
    main()
