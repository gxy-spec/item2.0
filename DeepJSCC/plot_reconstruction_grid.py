"""
Create a paper-style reconstruction grid for DeepJSCC.

Rows correspond to compression ratios and columns correspond to test SNRs.
The first column always shows the original image.

Usage:
    python plot_reconstruction_grid.py

Output:
    ./results/figures/reconstruction_grid.png
"""

import glob
import os

# Work around duplicate OpenMP runtime initialization in some Windows Python setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch

from data_load import load_cifar10_data
from model import Autoencoder
from utils import awgn_channel


MODEL_DIR = "./models"
SAVE_PATH = "./results/figures/reconstruction_grid.png"
TARGET_RATIOS = [0.05, 0.12, 0.30]
EXCLUDED_RATIO = 0.0833
TEST_SNRS = [0, 5, 10, 15, 20]
DATASET = "cifar10"
TRAIN_SNR = 10
CHANNEL_TYPE = "awgn"


def _select_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        print("Using CUDA for reconstruction.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS for reconstruction.")
        return torch.device("mps")
    print("Using CPU for reconstruction.")
    return torch.device("cpu")


def _parse_compression_ratio(model_path):
    """Parse compression ratio from model filename."""
    filename = os.path.basename(model_path)
    ratio_str = filename.split("_kn", maxsplit=1)[1].rsplit(".pth", maxsplit=1)[0]
    return float(ratio_str)


def _infer_latent_channels(compression_ratio, image_shape):
    """Infer latent channel count k from compression ratio and image size."""
    channels, height, width = image_shape
    latent_height, latent_width = height // 4, width // 4
    latent_channels = max(
        1,
        round(compression_ratio * (channels * height * width) / (latent_height * latent_width)),
    )
    return latent_channels


def _format_tensor_for_plot(image_tensor):
    """Convert a CHW tensor to an HWC numpy array and clamp it to [0, 1]."""
    image = image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return image


def load_models(model_dir=MODEL_DIR, target_ratios=TARGET_RATIOS):
    """Load target DeepJSCC models from the model directory.

    Returns:
        dict[float, Autoencoder]: Mapping from compression ratio to model.
    """
    model_paths = sorted(glob.glob(os.path.join(model_dir, "deep_jscc_cifar10_snr10_awgn_kn*.pth")))
    if not model_paths:
        raise FileNotFoundError(f"No matching model files found in {model_dir}")

    device = _select_device()
    models = {}
    available_paths = {}

    for model_path in model_paths:
        try:
            compression_ratio = _parse_compression_ratio(model_path)
        except (IndexError, ValueError):
            print(f"Skipping model with unrecognized name: {os.path.basename(model_path)}")
            continue

        if abs(compression_ratio - EXCLUDED_RATIO) < 1e-8:
            print(f"Skipping excluded model: {os.path.basename(model_path)}")
            continue

        if compression_ratio in target_ratios:
            available_paths[compression_ratio] = model_path

    missing_ratios = [ratio for ratio in target_ratios if ratio not in available_paths]
    if missing_ratios:
        raise FileNotFoundError(
            f"Missing required model files for compression ratios: {', '.join(f'{ratio:.2f}' for ratio in missing_ratios)}"
        )

    image_shape = (3, 32, 32)
    for compression_ratio in target_ratios:
        model_path = available_paths[compression_ratio]
        latent_channels = _infer_latent_channels(compression_ratio, image_shape)
        model = Autoencoder(k=latent_channels).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models[compression_ratio] = model
        print(f"Loaded model {os.path.basename(model_path)} -> CR={compression_ratio:.2f}, k={latent_channels}")

    return models, device


def get_test_image():
    """Get the first image from the CIFAR10 test set."""
    _, test_loader = load_cifar10_data(batch_size=1)
    image_tensor, _ = next(iter(test_loader))
    image_tensor = image_tensor[0]
    print(f"Loaded the first CIFAR10 test image for visualization.")
    return image_tensor


def reconstruct_image(model, image_tensor, snr_db, device):
    """Reconstruct one image by explicitly running encoder -> channel -> decoder."""
    input_batch = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_interleaved = model.encoder(input_batch)
        latent_complex = torch.complex(encoded_interleaved[:, 0::2], encoded_interleaved[:, 1::2])
        noisy_latent = awgn_channel(latent_complex, snr_db)

        decoder_input = torch.empty_like(encoded_interleaved)
        decoder_input[:, 0::2] = torch.real(noisy_latent)
        decoder_input[:, 1::2] = torch.imag(noisy_latent)

        reconstructed = model.decoder(decoder_input)[0]

    return reconstructed.clamp(0.0, 1.0)


def plot_reconstruction_grid(models, original_image, snr_list=TEST_SNRS, save_path=SAVE_PATH):
    """Plot a reconstruction grid over compression ratios and SNR values."""
    compression_ratios = sorted(models.keys())
    num_rows = len(compression_ratios)
    num_cols = 1 + len(snr_list)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 8))

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].set_title("Original")
    for col_index, snr_db in enumerate(snr_list, start=1):
        axes[0, col_index].set_title(f"SNR={snr_db}dB")

    original_plot = _format_tensor_for_plot(original_image)

    for row_index, compression_ratio in enumerate(compression_ratios):
        row_axes = axes[row_index]
        row_axes[0].imshow(original_plot)
        row_axes[0].text(
            -0.25,
            0.5,
            f"CR={compression_ratio:.2f}",
            transform=row_axes[0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )

        model = models[compression_ratio]
        device = next(model.parameters()).device

        for col_index, snr_db in enumerate(snr_list, start=1):
            reconstructed = reconstruct_image(model, original_image, snr_db, device)
            row_axes[col_index].imshow(_format_tensor_for_plot(reconstructed))

        for axis in row_axes:
            axis.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Reconstruction grid saved to {save_path}")


def main():
    """Load models, reconstruct one fixed test image, and save the grid."""
    print(
        f"Generating reconstruction grid for dataset={DATASET}, "
        f"train_snr={TRAIN_SNR}dB, channel={CHANNEL_TYPE}."
    )
    models, _ = load_models()
    original_image = get_test_image()
    plot_reconstruction_grid(models, original_image)


if __name__ == "__main__":
    main()
