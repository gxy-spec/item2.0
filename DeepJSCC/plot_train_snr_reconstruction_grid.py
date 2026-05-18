"""
Create a paper-style reconstruction grid under different training SNR settings.

Usage:
    python plot_train_snr_reconstruction_grid.py

Output:
    ./results/figures/train_snr_reconstruction_grid.png
"""

import os

import matplotlib.pyplot as plt

from train_snr_experiment_utils import (
    TEST_SNRS,
    format_tensor_for_plot,
    get_test_image,
    load_models,
    reconstruct_image,
)


SAVE_PATH = "./results/figures/train_snr_reconstruction_grid.png"


def plot_reconstruction_grid(models, original_image, save_path=SAVE_PATH, test_snrs=TEST_SNRS):
    """Plot a reconstruction grid over training SNR and test SNR."""
    train_snrs = sorted(models.keys())
    num_rows = len(train_snrs)
    num_cols = 1 + len(test_snrs)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 8))

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].set_title("Original")
    for col_index, test_snr in enumerate(test_snrs, start=1):
        axes[0, col_index].set_title(f"{test_snr}dB")

    original_plot = format_tensor_for_plot(original_image)

    for row_index, train_snr in enumerate(train_snrs):
        row_axes = axes[row_index]
        row_axes[0].imshow(original_plot)
        row_axes[0].text(
            -0.28,
            0.5,
            f"Train={train_snr}dB",
            transform=row_axes[0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )

        model = models[train_snr]
        device = next(model.parameters()).device

        for col_index, test_snr in enumerate(test_snrs, start=1):
            reconstructed = reconstruct_image(model, original_image, test_snr, device)
            row_axes[col_index].imshow(format_tensor_for_plot(reconstructed))

        for axis in row_axes:
            axis.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def main():
    """Load models, reconstruct one fixed test image, and save the grid."""
    models = load_models()
    original_image = get_test_image()
    plot_reconstruction_grid(models, original_image)


if __name__ == "__main__":
    main()
