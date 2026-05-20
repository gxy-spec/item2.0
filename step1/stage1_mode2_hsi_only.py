import json
import os
from datetime import datetime

# Work around duplicate OpenMP runtime initialization in some Windows setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class HSISingleStreamAE(nn.Module):
    """HSI-only autoencoder，支持信道噪声模拟、语义瓶颈及可选分类头。

    forward 返回 (reconstruction, logits) 其中 logits 在无分类头时为 None。
    """

    def __init__(self, input_dim: int, latent_dim: int, num_classes: int = 0):
        super(HSISingleStreamAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, input_dim),
        )
        self.num_classes = int(num_classes)
        if self.num_classes > 0:
            self.classifier = nn.Linear(latent_dim, self.num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor, snr_db=None, channel_type="awgn"):
        semantic = self.encoder(x)

        if snr_db is not None:
            signal_power = torch.mean(semantic ** 2)
            noise_variance = signal_power / (10 ** (snr_db / 10.0))
            if channel_type == "awgn":
                noise = torch.randn_like(semantic) * torch.sqrt(noise_variance + 1e-12)
                semantic = semantic + noise
            elif channel_type == "rayleigh":
                # 按样本生成幅度衰落因子（简化处理：同一样本所有语义通道使用相同衰落幅度）
                h_real = torch.randn(semantic.shape[0], 1, device=semantic.device) / np.sqrt(2)
                h_imag = torch.randn(semantic.shape[0], 1, device=semantic.device) / np.sqrt(2)
                h = torch.sqrt(h_real ** 2 + h_imag ** 2)
                noise = torch.randn_like(semantic) * torch.sqrt(noise_variance + 1e-12)
                semantic = h * semantic + noise
            else:
                raise ValueError(f"Unsupported channel type: {channel_type}")

        reconstruction = self.decoder(semantic)
        logits = self.classifier(semantic) if self.classifier is not None else None
        return reconstruction, logits


def load_mat_data(mat_path: str):
    """Load MATLAB data robustly from .mat file."""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    try:
        mat_contents = sio.loadmat(mat_path)
        keys = [key for key in mat_contents.keys() if not key.startswith("_")]
        if not keys:
            raise KeyError("No valid MATLAB variable keys found in file.")
        return {key: mat_contents[key] for key in keys}
    except Exception as exc:
        raise RuntimeError(f"Failed to load MAT file {mat_path}: {exc}")


def normalize_hsi_array(hsi_array: np.ndarray) -> np.ndarray:
    """Normalize HSI array to shape (600, 166, 20) and select first 20 bands."""
    arr = np.asarray(hsi_array)
    if arr.ndim != 3:
        raise ValueError(f"HSI array must be 3D, got shape {arr.shape}.")

    if arr.shape[:2] == (166, 600):
        arr = np.transpose(arr, (1, 0, 2))
    elif arr.shape[:2] == (600, 166):
        pass
    else:
        raise ValueError(f"Unexpected HSI spatial shape {arr.shape[:2]}, expected (600,166) or (166,600).")

    if arr.shape[2] < 20:
        raise ValueError(f"HSI band dimension must be at least 20, got {arr.shape[2]}.")

    if arr.shape[2] != 20:
        arr = arr[:, :, :20]
        print(f"HSI band count is {hsi_array.shape[2]}; using first 20 bands for mode2.")

    if arr.shape != (600, 166, 20):
        raise ValueError(f"After normalization, HSI shape must be (600,166,20), got {arr.shape}.")

    return arr.astype(np.float32)


def normalize_gt_array(gt_array: np.ndarray) -> np.ndarray:
    """Normalize GT array to shape (600, 166)."""
    arr = np.asarray(gt_array)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"GT array must be 2D, got shape {arr.shape}.")

    if arr.shape == (166, 600):
        arr = arr.T
    if arr.shape != (600, 166):
        raise ValueError(f"Normalized GT shape must be (600,166), got {arr.shape}.")

    return arr.astype(np.int32)


def load_italy_hsi_dataset():
    """Load Italy HSI and GT maps, then extract valid HSI pixels."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hsi_path = os.path.join(current_dir, "Italy_hsi.mat")
    gt_path = os.path.join(current_dir, "Italy_gt.mat")
    fallback_gt_path = os.path.join(current_dir, "allgrd.mat")

    raw_hsi = load_mat_data(hsi_path)
    if os.path.exists(gt_path):
        raw_gt = load_mat_data(gt_path)
        print("Using Italy_gt.mat for ground truth.")
    else:
        raw_gt = load_mat_data(fallback_gt_path)
        print("Italy_gt.mat not found, using allgrd.mat as fallback ground truth.")

    hsi_key = [key for key in raw_hsi.keys() if not key.startswith("_")][0]
    gt_key = [key for key in raw_gt.keys() if not key.startswith("_")][0]

    hsi_map = normalize_hsi_array(raw_hsi[hsi_key])
    gt_map = normalize_gt_array(raw_gt[gt_key])

    hsi_flat = hsi_map.reshape(-1, hsi_map.shape[2])
    gt_flat = gt_map.reshape(-1)
    valid_mask_flat = gt_flat > 0
    valid_hsi = hsi_flat[valid_mask_flat].astype(np.float32)
    valid_labels = (gt_flat[valid_mask_flat] - 1).astype(np.int64)
    num_classes = int(valid_labels.max()) + 1 if valid_labels.size > 0 else 0
    valid_mask = valid_mask_flat.reshape(hsi_map.shape[0], hsi_map.shape[1])

    print(f"HSI shape: {hsi_map.shape}")
    print(f"GT shape: {gt_map.shape}")
    print(f"Valid labeled pixels: {valid_hsi.shape[0]}")

    return hsi_map, gt_map, valid_mask, valid_hsi, valid_labels, num_classes


def normalize_by_train_stats(train_array: np.ndarray, test_array: np.ndarray):
    mean = train_array.mean(axis=0, keepdims=True)
    std = train_array.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    train_norm = ((train_array - mean) / std).astype(np.float32)
    test_norm = ((test_array - mean) / std).astype(np.float32)
    stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return train_norm, test_norm, stats


def build_dataloader(valid_hsi: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(valid_hsi))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def compute_nmse(pred: np.ndarray, target: np.ndarray) -> float:
    num = np.sum((pred - target) ** 2)
    den = np.sum(target ** 2) + 1e-12
    return float(num / den)


def evaluate_at_snr(
    model: HSISingleStreamAE,
    valid_norm: np.ndarray,
    valid_orig: np.ndarray,
    valid_labels: np.ndarray,
    hsi_stats: dict,
    device: torch.device,
    snr_db: float,
    channel_type: str = "awgn",
    num_repeats: int = 5,
    batch_size: int = 512,
) -> dict:
    """Evaluate HSI NMSE and classification accuracy under specified SNR and channel."""
    model.eval()
    nmse_list = []
    acc_list = []

    for _ in range(num_repeats):
        total_num = 0.0
        total_den = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(valid_norm), batch_size):
                end = min(len(valid_norm), start + batch_size)
                batch_norm = torch.from_numpy(valid_norm[start:end]).float().to(device)
                batch_orig = valid_orig[start:end]
                batch_labels = valid_labels[start:end]

                recon_norm, logits = model(batch_norm, snr_db=snr_db, channel_type=channel_type)
                recon_orig = recon_norm.cpu().numpy() * hsi_stats["std"] + hsi_stats["mean"]

                total_num += np.sum((recon_orig - batch_orig) ** 2)
                total_den += np.sum(batch_orig ** 2)

                if logits is not None and batch_labels is not None and batch_labels.size > 0:
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    correct += int((preds == batch_labels).sum())
                    total += int(batch_labels.shape[0])

        nmse_list.append(total_num / (total_den + 1e-12))
        acc_list.append(correct / max(total, 1))

    return {
        "hsi_nmse": float(np.mean(nmse_list)),
        "hsi_nmse_std": float(np.std(nmse_list)),
        "accuracy": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list)),
    }


def save_metric_curves(results_by_dim, snr_values, output_path, channel_type="awgn"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for dim, metrics in results_by_dim.items():
        axes[0].plot(snr_values, metrics["hsi_nmse"], marker="o", linewidth=2, label=f"Semantic Dim={dim}")
        axes[2].plot(snr_values, metrics.get("accuracy", []), marker="^", linewidth=2, label=f"Semantic Dim={dim}")

        axes[0].fill_between(
            snr_values,
            np.array(metrics["hsi_nmse"]) - np.array(metrics.get("hsi_nmse_std", np.zeros_like(metrics["hsi_nmse"]))),
            np.array(metrics["hsi_nmse"]) + np.array(metrics.get("hsi_nmse_std", np.zeros_like(metrics["hsi_nmse"]))),
            alpha=0.15,
        )
        axes[2].fill_between(
            snr_values,
            np.array(metrics.get("accuracy", [])) - np.array(metrics.get("accuracy_std", np.zeros_like(metrics.get("accuracy", [])))),
            np.array(metrics.get("accuracy", [])) + np.array(metrics.get("accuracy_std", np.zeros_like(metrics.get("accuracy", [])))),
            alpha=0.15,
        )

    axes[0].set_title("HSI Reconstruction Error vs. SNR")
    axes[0].set_xlabel("Received SNR (dB)")
    axes[0].set_ylabel("Normalised Mean Squared Error (NMSE)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("LiDAR Reconstruction Error vs. SNR")
    axes[1].set_xlabel("Received SNR (dB)")
    axes[1].set_ylabel("Normalised Mean Squared Error (NMSE)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    axes[2].set_title("Classification Accuracy vs. SNR")
    axes[2].set_xlabel("Received SNR (dB)")
    axes[2].set_ylabel("Classification Accuracy")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.suptitle(
        f"{channel_type.upper()} Channel Performance: Reconstruction and Classification Metrics",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved metric curve to {output_path}")


def reconstruct_full_maps(
    model: HSISingleStreamAE,
    hsi_map: np.ndarray,
    hsi_stats: dict,
    device: torch.device,
    snr_db: float,
    channel_type: str = "awgn",
    batch_size: int = 512,
) -> np.ndarray:
    height, width, hsi_dim = hsi_map.shape
    flat_hsi = hsi_map.reshape(-1, hsi_dim).astype(np.float32)
    normalized_flat = (flat_hsi - hsi_stats["mean"]) / hsi_stats["std"]

    reconstructed_list = []
    model.eval()
    with torch.no_grad():
        for start in range(0, normalized_flat.shape[0], batch_size):
            batch = torch.from_numpy(normalized_flat[start:start + batch_size]).float().to(device)
            out = model(batch, snr_db=snr_db, channel_type=channel_type)
            # model may return (reconstruction, logits)
            recon_norm = out[0] if isinstance(out, tuple) else out
            reconstructed_list.append(recon_norm.cpu().numpy())

    reconstructed_flat = np.vstack(reconstructed_list)
    reconstructed_orig = reconstructed_flat * hsi_stats["std"] + hsi_stats["mean"]
    return reconstructed_orig.reshape(height, width, hsi_dim)


def save_reconstruction_figure(
    hsi_map,
    reconstructions,
    visual_dims,
    output_path,
    snr_db,
    channel_type="awgn",
    valid_mask=None,
):
    # Align layout and style with mode1: 2 rows x (1 + len(visual_dims)) cols
    band_idx = min(10, hsi_map.shape[2] - 1)
    hsi_vmin = float(hsi_map[:, :, band_idx].min())
    hsi_vmax = float(hsi_map[:, :, band_idx].max())

    original_hsi = hsi_map[:, :, band_idx].copy()
    if valid_mask is not None:
        original_hsi = np.where(valid_mask, original_hsi, np.nan)

    ncols = 1 + len(visual_dims)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    # Left-most: original HSI band (top row equivalent)
    axes[0].imshow(original_hsi, cmap="jet", vmin=hsi_vmin, vmax=hsi_vmax)
    axes[0].set_title(f"Original HSI\n(Band {band_idx})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Following columns: reconstructed HSI bands (one per semantic dim)
    for idx, dim in enumerate(visual_dims):
        recon_hsi = reconstructions[dim][:, :, band_idx].copy()
        if valid_mask is not None:
            recon_hsi = np.where(valid_mask, recon_hsi, np.nan)
        axes[idx + 1].imshow(recon_hsi, cmap="jet", vmin=hsi_vmin, vmax=hsi_vmax)
        axes[idx + 1].set_title(f"Reconstructed HSI\n(Semantic Dim = {dim})", fontsize=11)
        axes[idx + 1].axis("off")

    plt.suptitle(
        f"Reconstruction Comparison under {channel_type.upper()} Channel (SNR={snr_db} dB)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved reconstruction figure to {output_path}")


def train_autoencoder(
    model: HSISingleStreamAE,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    """Train the HSI-only autoencoder with MSE loss."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            reconstruction = out[0] if isinstance(out, tuple) else out
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{epochs:02d} | MSE Loss: {average_loss:.6f}")


def reconstruct_full_hsi(
    model: HSISingleStreamAE,
    full_hsi: np.ndarray,
    hsi_mean: np.ndarray,
    hsi_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Reconstruct the full HSI image in safe batches and re-scale to original units."""
    model.eval()
    flat_hsi = full_hsi.reshape(-1, full_hsi.shape[2]).astype(np.float32)
    normalized_flat = (flat_hsi - hsi_mean) / hsi_std

    reconstructed_chunks = []
    with torch.no_grad():
        for start in range(0, normalized_flat.shape[0], batch_size):
            batch = torch.from_numpy(normalized_flat[start:start + batch_size]).to(device)
            out = model(batch)
            reconstructed = out[0] if isinstance(out, tuple) else out
            reconstructed_chunks.append(reconstructed.cpu().numpy())

    reconstructed_flat = np.vstack(reconstructed_chunks)
    reconstructed_full = reconstructed_flat * hsi_std + hsi_mean
    return reconstructed_full.reshape(full_hsi.shape)


def plot_hsi_reconstructions(
    original_hsi: np.ndarray,
    reconstructions: dict,
    visual_dims: list,
    output_path: str,
):
    """Save a comparison figure for original and reconstructed HSI band 10."""
    band_index = 9
    original_band = original_hsi[:, :, band_index]
    vmin = float(original_band.min())
    vmax = float(original_band.max())

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    titles = ["Original HSI Band 10"] + [f"Reconstructed HSI Band 10\n(latent_dim={dim})" for dim in visual_dims]
    images = [original_band] + [recon[:, :, band_index] for recon in reconstructions.values()]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Mode 2: HSI-only Semantic Autoencoder Reconstruction Comparison",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved perception analysis figure to {output_path}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hsi_map, gt_map, valid_mask, valid_hsi, valid_labels, num_classes = load_italy_hsi_dataset()
    hsi_dim = hsi_map.shape[2]

    batch_size = 128
    epochs = 30
    learning_rate = 0.001
    train_snr_db = 10
    channel_type = "rayleigh"
    snr_values = [-5, 0, 5, 10, 15, 20]
    visual_dims = [2, 6, 12]
    eval_repeats = 5
    cls_loss_weight = 1.0

    # Split valid labeled pixels into train/test for supervised classification
    x_train, x_test, y_train, y_test = train_test_split(
        valid_hsi, valid_labels, test_size=0.3, random_state=42, stratify=valid_labels
    )
    x_train_norm, x_test_norm, hsi_stats = normalize_by_train_stats(x_train, x_test)

    train_dataset = TensorDataset(torch.from_numpy(x_train_norm), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test_norm), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("====================================================================")
    print(" 正在启动『模式二：单模态纯 HSI 语义自编码网络』实验")
    print("====================================================================")
    print(f"Device: {device}")
    print(f"HSI map shape: {hsi_map.shape}")
    print(f"Valid labeled pixels: {valid_hsi.shape[0]}")
    print(f"Latent dims: {visual_dims}")
    print(f"Channel type: {channel_type}")
    print(f"Train SNR: {train_snr_db} dB")

    results_by_dim = {}
    reconstructions = {}


    for latent_dim in visual_dims:
        print(f"\n[训练中] HSI-only 自编码器语义维度: {latent_dim}")
        model = HSISingleStreamAE(input_dim=hsi_dim, latent_dim=latent_dim, num_classes=num_classes).to(device)
        rec_criterion = nn.MSELoss()
        cls_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                recon_norm, logits = model(batch_x, snr_db=train_snr_db, channel_type=channel_type)
                loss_rec = rec_criterion(recon_norm, batch_x)
                loss_cls = cls_criterion(logits, batch_y)
                loss = loss_rec + cls_loss_weight * loss_cls
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"  Epoch [{epoch + 1}/{epochs}] | Joint Loss: {epoch_loss / len(train_loader):.6f}")

        model_name = f"mode2_dim{latent_dim}_trainSNR{train_snr_db}dB_{channel_type}_{timestamp}.pth"
        torch.save(model.state_dict(), os.path.join(output_dir, model_name))
        print(f"  -> 模型已保存: {model_name}")

        metrics = {
            "hsi_nmse": [],
            "hsi_nmse_std": [],
            "accuracy": [],
            "accuracy_std": [],
        }

        for snr_db in snr_values:
            eval_result = evaluate_at_snr(
                model,
                x_test_norm,
                x_test,
                y_test,
                hsi_stats,
                device,
                snr_db,
                channel_type=channel_type,
                num_repeats=eval_repeats,
            )
            metrics["hsi_nmse"].append(eval_result["hsi_nmse"])
            metrics["hsi_nmse_std"].append(eval_result["hsi_nmse_std"])
            metrics["accuracy"].append(eval_result["accuracy"])
            metrics["accuracy_std"].append(eval_result["accuracy_std"])
            print(
                f"  SNR={snr_db:>3} dB | HSI NMSE={eval_result['hsi_nmse']:.6f}±{eval_result['hsi_nmse_std']:.6f} | Accuracy={eval_result['accuracy']:.4f}±{eval_result['accuracy_std']:.4f}"
            )

        results_by_dim[latent_dim] = metrics
        recon = reconstruct_full_maps(
            model,
            hsi_map,
            hsi_stats,
            device,
            snr_db=train_snr_db,
            channel_type=channel_type,
        )
        reconstructions[latent_dim] = recon

    metrics_fig_name = f"mode2_metrics_trainSNR{train_snr_db}dB_{channel_type}_{timestamp}.png"
    recon_fig_name = f"mode2_reconstruction_trainSNR{train_snr_db}dB_{channel_type}_{timestamp}.png"
    metrics_json_name = f"mode2_metrics_trainSNR{train_snr_db}dB_{channel_type}_{timestamp}.json"

    save_metric_curves(
        results_by_dim,
        snr_values,
        os.path.join(output_dir, metrics_fig_name),
        channel_type=channel_type,
    )
    save_reconstruction_figure(
        hsi_map,
        reconstructions,
        visual_dims,
        os.path.join(output_dir, recon_fig_name),
        snr_db=train_snr_db,
        channel_type=channel_type,
        valid_mask=valid_mask,
    )

    with open(os.path.join(output_dir, metrics_json_name), "w", encoding="utf-8") as file:
        json.dump(
            {
                "timestamp": timestamp,
                "train_snr_db": train_snr_db,
                "channel_type": channel_type,
                "snr_values": snr_values,
                "visual_dims": visual_dims,
                "eval_repeats": eval_repeats,
                "results": results_by_dim,
                "normalization": {
                    "mean_shape": list(hsi_stats["mean"].shape),
                    "std_shape": list(hsi_stats["std"].shape),
                },
            },
            file,
            indent=2,
        )

    print("\n[OK] 全部实验完成。")
    print(f"输出目录: {output_dir}")
    print(f"指标图: {metrics_fig_name}")
    print(f"重构图: {recon_fig_name}")
    print(f"指标JSON: {metrics_json_name}")


if __name__ == "__main__":
    main()
