"""
Utility helpers for the stage-2 training-SNR robustness plots.
"""

import glob
import json
import os
import re

# Work around duplicate OpenMP runtime initialization in some Windows Python setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from data_load import load_cifar10_data
from model import Autoencoder
from utils import awgn_channel


RESULTS_DIR = "./results"
MODEL_DIR = "./models"
TARGET_TRAIN_SNRS = [0, 10, 20]
TARGET_COMPRESSION_RATIO = 0.12
TEST_SNRS = [0, 5, 10, 15, 20]

_EVAL_PATTERN = re.compile(
    r"^evaluation_deep_jscc_cifar10_snr(?P<train_snr>\d+)_awgn_kn(?P<ratio>\d+\.\d+)\.json$"
)
_MODEL_PATTERN = re.compile(
    r"^deep_jscc_cifar10_snr(?P<train_snr>\d+)_awgn_kn(?P<ratio>\d+\.\d+)\.pth$"
)


def _select_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        print("Using CUDA.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS.")
        return torch.device("mps")
    print("Using CPU.")
    return torch.device("cpu")


def _infer_latent_channels(compression_ratio, image_shape):
    """Infer the autoencoder latent channel count from the compression ratio."""
    channels, height, width = image_shape
    latent_height, latent_width = height // 4, width // 4
    latent_channels = max(
        1,
        round(compression_ratio * (channels * height * width) / (latent_height * latent_width)),
    )
    return latent_channels


def _parse_eval_filename(filename):
    """Parse train SNR and compression ratio from an evaluation filename."""
    match = _EVAL_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unrecognized evaluation filename: {filename}")
    train_snr = int(match.group("train_snr"))
    compression_ratio = float(match.group("ratio"))
    return train_snr, compression_ratio


def _parse_model_filename(filename):
    """Parse train SNR and compression ratio from a model filename."""
    match = _MODEL_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unrecognized model filename: {filename}")
    train_snr = int(match.group("train_snr"))
    compression_ratio = float(match.group("ratio"))
    return train_snr, compression_ratio


def load_results(results_dir=RESULTS_DIR, metric="psnr"):
    """Load AWGN metric curves for train_snr={0,10,20} at compression_ratio=0.12."""
    results = {}
    json_files = sorted(glob.glob(os.path.join(results_dir, "evaluation_*.json")))

    for json_file in json_files:
        filename = os.path.basename(json_file)

        try:
            train_snr, compression_ratio = _parse_eval_filename(filename)
        except ValueError:
            continue

        if train_snr not in TARGET_TRAIN_SNRS:
            continue
        if abs(compression_ratio - TARGET_COMPRESSION_RATIO) > 1e-8:
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable JSON file {filename}: {exc}")
            continue

        awgn_metric = data.get("awgn", {}).get(metric)
        if not isinstance(awgn_metric, dict):
            print(f"Skipping file without awgn->{metric} data: {filename}")
            continue

        try:
            snr_metric = {float(snr): float(value) for snr, value in awgn_metric.items()}
        except (TypeError, ValueError) as exc:
            print(f"Skipping invalid awgn->{metric} data in {filename}: {exc}")
            continue

        results[train_snr] = dict(sorted(snr_metric.items()))
        print(f"Loaded {filename} -> Train SNR={train_snr}dB")

    missing_train_snrs = [snr for snr in TARGET_TRAIN_SNRS if snr not in results]
    if missing_train_snrs:
        print(
            "Warning: missing evaluation results for train_snr="
            + ", ".join(f"{snr}dB" for snr in missing_train_snrs)
        )

    return dict(sorted(results.items()))


def load_models(model_dir=MODEL_DIR):
    """Load train_snr={0,10,20} models at compression_ratio=0.12."""
    model_files = sorted(glob.glob(os.path.join(model_dir, "deep_jscc_cifar10_snr*_awgn_kn*.pth")))
    device = _select_device()
    models = {}
    image_shape = (3, 32, 32)

    for model_file in model_files:
        filename = os.path.basename(model_file)

        try:
            train_snr, compression_ratio = _parse_model_filename(filename)
        except ValueError:
            continue

        if train_snr not in TARGET_TRAIN_SNRS:
            continue
        if abs(compression_ratio - TARGET_COMPRESSION_RATIO) > 1e-8:
            continue

        latent_channels = _infer_latent_channels(compression_ratio, image_shape)
        model = Autoencoder(k=latent_channels).to(device)
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models[train_snr] = model
        print(f"Loaded {filename} -> Train SNR={train_snr}dB, k={latent_channels}")

    missing_train_snrs = [snr for snr in TARGET_TRAIN_SNRS if snr not in models]
    if missing_train_snrs:
        raise FileNotFoundError(
            "Missing required model files for train_snr="
            + ", ".join(f"{snr}dB" for snr in missing_train_snrs)
        )

    return dict(sorted(models.items()))


def reconstruct_image(model, image_tensor, test_snr, device):
    """Run image -> encoder -> AWGN channel -> decoder for one image."""
    input_batch = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_interleaved = model.encoder(input_batch)
        latent_complex = torch.complex(encoded_interleaved[:, 0::2], encoded_interleaved[:, 1::2])
        noisy_latent = awgn_channel(latent_complex, test_snr)

        decoder_input = torch.empty_like(encoded_interleaved)
        decoder_input[:, 0::2] = torch.real(noisy_latent)
        decoder_input[:, 1::2] = torch.imag(noisy_latent)

        reconstructed = model.decoder(decoder_input)[0]

    return reconstructed.clamp(0.0, 1.0)


def get_test_image():
    """Get the first sample from the CIFAR10 test set."""
    _, test_loader = load_cifar10_data(batch_size=1)
    image_tensor, _ = next(iter(test_loader))
    print("Loaded the first CIFAR10 test image.")
    return image_tensor[0]


def format_tensor_for_plot(image_tensor):
    """Convert a CHW tensor to a clamped HWC numpy image."""
    return image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
