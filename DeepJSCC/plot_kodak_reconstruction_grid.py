"""
Visualize Kodak reconstruction results for one fixed image.

Usage:
    python plot_kodak_reconstruction_grid.py

Output:
    ./results/figures/kodak_reconstruction_grid_kn0.0833.png
"""

import os

# Work around duplicate OpenMP runtime initialization in some Windows Python setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch

from data_load import load_kodak_dataset
from model import Autoencoder
from utils import awgn_channel


MODEL_PATH = "./models/deep_jscc_cifar10_snr10_awgn_kn0.0833.pth"
KODAK_DIR = "./kodak_dataset"
SAVE_PATH = "./results/figures/kodak_reconstruction_grid_kn0.0833.png"
TEST_SNRS = [0, 5, 10, 15, 20]
COMPRESSION_RATIO = 0.0833


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
    return image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def load_model(model_path=MODEL_PATH):
    """Load the DeepJSCC model for Kodak visualization."""
    device = _select_device()
    test_loader = load_kodak_dataset(path=KODAK_DIR, batch_size=1)
    image_tensor, _ = next(iter(test_loader))
    image_shape = tuple(image_tensor[0].shape)
    latent_channels = _infer_latent_channels(COMPRESSION_RATIO, image_shape)

    model = Autoencoder(k=latent_channels).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Inferred k={latent_channels} for Kodak image shape {image_shape}")
    return model, device


def get_test_image():
    """Get the first Kodak image without randomness."""
    test_loader = load_kodak_dataset(path=KODAK_DIR, batch_size=1)
    image_tensor, _ = next(iter(test_loader))
    print("Loaded the first Kodak image for visualization.")
    return image_tensor[0]


def reconstruct_image(model, image_tensor, snr_db, device):
    """Reconstruct one image by running encoder -> channel -> decoder."""
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


def plot_reconstruction_grid(model, original_image, device, save_path=SAVE_PATH):
    """Plot the original Kodak image and reconstructions at multiple SNR values."""
    num_cols = 1 + len(TEST_SNRS)
    fig, axes = plt.subplots(1, num_cols, figsize=(14, 4))

    axes[0].imshow(_format_tensor_for_plot(original_image))
    axes[0].set_title("Original")
    axes[0].axis("off")

    for index, snr_db in enumerate(TEST_SNRS, start=1):
        reconstructed = reconstruct_image(model, original_image, snr_db, device)
        axes[index].imshow(_format_tensor_for_plot(reconstructed))
        axes[index].set_title(f"SNR={snr_db}dB")
        axes[index].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def main():
    """Load the model and generate the Kodak reconstruction comparison figure."""
    model, device = load_model()
    original_image = get_test_image()
    plot_reconstruction_grid(model, original_image, device)


if __name__ == "__main__":
    main()
