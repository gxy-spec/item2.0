import json
import os
import math
from datetime import datetime

# Work around duplicate OpenMP runtime initialization in some Windows setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class HSISingleStreamAE(nn.Module):
    def __init__(self, hsi_dim, hsi_sem, num_classes):
        super().__init__()

        # 编码器：映射到语义特征空间
        self.encoder = nn.Sequential(
            nn.Linear(hsi_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, hsi_sem),
        )

        # 分类器与解码器
        self.classifier = nn.Linear(hsi_sem, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(hsi_sem, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, hsi_dim),
        )

    def forward(self, x, snr_db=None, channel_type="awgn"):
        semantic = self.encoder(x)

        if snr_db is not None:
            # 1. 严格计算发送信号的真实平均功率
            signal_power = torch.mean(semantic ** 2)
            noise_variance = signal_power / (10 ** (snr_db / 10.0))

            if channel_type == "awgn":
                noise = torch.randn_like(semantic) * torch.sqrt(noise_variance + 1e-12)
                semantic = semantic + noise

            elif channel_type == "rayleigh":
                # 2. 模拟严格的复数基带瑞利衰落信道（对齐模态1的纯实数幅度衰落模型）
                h_real = torch.randn(semantic.shape[0], 1, device=semantic.device) / math.sqrt(2)
                h_imag = torch.randn(semantic.shape[0], 1, device=semantic.device) / math.sqrt(2)
                h_mag = torch.sqrt(h_real ** 2 + h_imag ** 2)  # 严格的瑞利分布幅度
                
                # 产生噪声
                noise = torch.randn_like(semantic) * torch.sqrt(noise_variance + 1e-12)
                
                # 信道传输过后的接收信号
                y_received = h_mag * semantic + noise
                
                # 3. 接收端引入迫零均衡 (Zero-Forcing Equalization)
                # 使用平滑保护项 epsilon，完全对齐模态1，防止噪声暴涨
                epsilon = 1e-2
                semantic = y_received * h_mag / (h_mag ** 2 + epsilon)

            else:
                raise ValueError(f"Unsupported channel type: {channel_type}")

        logits = self.classifier(semantic)
        hsi_hat = self.decoder(semantic)
        return hsi_hat, logits


def load_trento_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_hsi = sio.loadmat(os.path.join(current_dir, "Italy_hsi.mat"))
    raw_gt = sio.loadmat(os.path.join(current_dir, "allgrd.mat"))

    hsi_key = [key for key in raw_hsi.keys() if not key.startswith("_")][0]
    gt_key = [key for key in raw_gt.keys() if not key.startswith("_")][0]

    hsi_map = raw_hsi[hsi_key]
    gt_map = raw_gt[gt_key]

    height, width, hsi_dim = hsi_map.shape
    labels_flat = gt_map.reshape(-1)
    valid_idx = np.where(labels_flat > 0)[0]
    valid_mask = gt_map > 0

    hsi_flat = hsi_map.reshape(-1, hsi_dim)
    labels = (labels_flat[valid_idx] - 1).astype(np.int64)

    x_hsi = hsi_flat[valid_idx].astype(np.float32)
    num_classes = int(labels.max()) + 1

    print(f"HSI shape: {hsi_map.shape}")
    print(f"GT shape: {gt_map.shape}")
    print(f"Valid labeled pixels: {len(valid_idx)}")
    print(f"Number of classes: {num_classes}")

    return (
        x_hsi,
        labels,
        height,
        width,
        hsi_dim,
        num_classes,
        hsi_map,
        valid_mask,
    )


def normalize_by_train_stats(train_array, test_array):
    mean = train_array.mean(axis=0, keepdims=True)
    std = train_array.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    train_norm = ((train_array - mean) / std).astype(np.float32)
    test_norm = ((test_array - mean) / std).astype(np.float32)
    stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return train_norm, test_norm, stats


def create_dataloaders(x_hsi, labels, batch_size):
    x_hsi_train, x_hsi_test, y_train, y_test = train_test_split(
        x_hsi,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    x_hsi_train_norm, x_hsi_test_norm, hsi_stats = normalize_by_train_stats(x_hsi_train, x_hsi_test)

    train_dataset = TensorDataset(
        torch.from_numpy(x_hsi_train_norm),
        torch.from_numpy(y_train),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_hsi_test_norm),
        torch.from_numpy(y_test),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, hsi_stats, x_hsi_test_norm, y_test


def evaluate_at_snr(model, test_loader, test_x_norm, test_y, device, snr_db, channel_type="awgn", num_repeats=5):
    aggregate = {"hsi_nmse": [], "accuracy": []}
    model.eval()

    for _ in range(num_repeats):
        hsi_nmse_num = 0.0
        hsi_nmse_den = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_hsi, batch_labels in test_loader:
                batch_hsi = batch_hsi.to(device)
                batch_labels = batch_labels.to(device)

                hsi_hat, logits = model(batch_hsi, snr_db=snr_db, channel_type=channel_type)
                
                # 【关键修复】直接在归一化空间计算特征层面的 NMSE，完全等价于模态1的计算基准
                hsi_nmse_num += torch.sum((hsi_hat - batch_hsi) ** 2).item()
                hsi_nmse_den += torch.sum(batch_hsi ** 2).item()

                predicted = torch.argmax(logits, dim=1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        aggregate["hsi_nmse"].append(hsi_nmse_num / (hsi_nmse_den + 1e-12))
        aggregate["accuracy"].append(correct / max(total, 1))

    return {
        "hsi_nmse": float(np.mean(aggregate["hsi_nmse"])),
        "accuracy": float(np.mean(aggregate["accuracy"])),
        "hsi_nmse_std": float(np.std(aggregate["hsi_nmse"])),
        "accuracy_std": float(np.std(aggregate["accuracy"])),
    }


def save_metric_curves(results_by_dim, snr_values, output_path, channel_type="awgn"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for dim, metrics in results_by_dim.items():
        axes[0].plot(snr_values, metrics["hsi_nmse"], marker="o", linewidth=2, label=f"Semantic Dim={dim}")
        axes[1].plot(snr_values, metrics["accuracy"], marker="^", linewidth=2, label=f"Semantic Dim={dim}")

        axes[0].fill_between(
            snr_values,
            np.array(metrics["hsi_nmse"]) - np.array(metrics["hsi_nmse_std"]),
            np.array(metrics["hsi_nmse"]) + np.array(metrics["hsi_nmse_std"]),
            alpha=0.15,
        )
        axes[1].fill_between(
            snr_values,
            np.array(metrics["accuracy"]) - np.array(metrics["accuracy_std"]),
            np.array(metrics["accuracy"]) + np.array(metrics["accuracy_std"]),
            alpha=0.15,
        )

    axes[0].set_title("HSI Reconstruction Error vs. SNR")
    axes[0].set_xlabel("Received SNR (dB)")
    axes[0].set_ylabel("Normalised Mean Squared Error (NMSE)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("Classification Accuracy vs. SNR")
    axes[1].set_xlabel("Received SNR (dB)")
    axes[1].set_ylabel("Classification Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.suptitle(
        f"{channel_type.upper()} Channel Performance: Model 2 (Single Stream HSI)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def reconstruct_full_maps(model, hsi_map, device, snr_db, channel_type="awgn", hsi_stats=None):
    height, width, hsi_dim = hsi_map.shape

    hsi_all = hsi_map.reshape(-1, hsi_dim).astype(np.float32)
    hsi_all = (hsi_all - hsi_stats["mean"]) / hsi_stats["std"]

    hsi_hat_list = []

    model.eval()
    with torch.no_grad():
        for index in range(0, len(hsi_all), 512):
            chunk_h = torch.from_numpy(hsi_all[index:index + 512]).float().to(device)

            if chunk_h.shape[0] <= 1:
                h_hat = chunk_h
            else:
                h_hat, _ = model(
                    chunk_h,
                    snr_db=snr_db,
                    channel_type=channel_type,
                )

            h_hat_np = h_hat.cpu().numpy() * hsi_stats["std"] + hsi_stats["mean"]
            hsi_hat_list.append(h_hat_np)

    return {
        "hsi": np.vstack(hsi_hat_list).reshape(height, width, hsi_dim),
    }


def save_reconstruction_figure(
    hsi_map,
    reconstructions,
    visual_dims,
    output_path,
    snr_db,
    channel_type="awgn",
    valid_mask=None,
):
    band_idx = min(10, hsi_map.shape[2] - 1)
    hsi_vmin = float(hsi_map[:, :, band_idx].min())
    hsi_vmax = float(hsi_map[:, :, band_idx].max())

    original_hsi = hsi_map[:, :, band_idx].copy()
    if valid_mask is not None:
        original_hsi = np.where(valid_mask, original_hsi, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(original_hsi, cmap="jet", vmin=hsi_vmin, vmax=hsi_vmax)
    axes[0].set_title(f"Original HSI\n(Band {band_idx})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for idx, dim in enumerate(visual_dims):
        recon_hsi = reconstructions[dim]["hsi"][:, :, band_idx].copy()
        if valid_mask is not None:
            recon_hsi = np.where(valid_mask, recon_hsi, np.nan)
        axes[idx + 1].imshow(recon_hsi, cmap="jet", vmin=hsi_vmin, vmax=hsi_vmax)
        axes[idx + 1].set_title(f"Reconstructed HSI\n(Semantic Dim = {dim})", fontsize=11)
        axes[idx + 1].axis("off")

    plt.suptitle(
        f"Model 2 Reconstruction under {channel_type.upper()} Channel (SNR={snr_db} dB)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        x_hsi,
        labels,
        height,
        width,
        hsi_dim,
        num_classes,
        hsi_map,
        valid_mask,
    ) = load_trento_data()

    batch_size = 512
    epochs = 30
    learning_rate = 0.005
    train_snr_db = 10
    channel_type = "awgn"  # 改为升级版带有ZF均衡的真实瑞利信道
    snr_values = [-5, 0, 5, 10, 15, 20]
    visual_dims = [2, 6, 12]  # 分别对应模态1中两流之和得到的总特征维度
    eval_repeats = 5
    cls_loss_weight = 1.0
    show_valid_region_only = False

    train_loader, test_loader, hsi_stats, test_x_norm, test_y = create_dataloaders(x_hsi, labels, batch_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    results_by_dim = {}
    reconstructions = {}

    print("====================================================================")
    print(" 正在启动『单一模态 HSI』语义重构 + 分类 + 信道鲁棒性实验")
    print("====================================================================")

    for total_dim in visual_dims:
        print(f"\n[训练中] 语义压缩层总维度: {total_dim} (单模态高光谱)")

        model = HSISingleStreamAE(hsi_dim, total_dim, num_classes).to(device)
        criterion_rec = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_hsi, batch_labels in train_loader:
                batch_hsi = batch_hsi.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                hsi_hat, logits = model(
                    batch_hsi,
                    snr_db=train_snr_db,
                    channel_type=channel_type,
                )

                loss_hsi = criterion_rec(hsi_hat, batch_hsi)
                loss_cls = criterion_cls(logits, batch_labels)
                loss = loss_hsi + cls_loss_weight * loss_cls

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"  Epoch [{epoch + 1}/{epochs}] | Joint Loss: {epoch_loss / len(train_loader):.6f}")

        model_name = f"mode2_dim{total_dim}_trainSNR{train_snr_db}dB_{channel_type}_{timestamp}.pth"
        torch.save(model.state_dict(), os.path.join(output_dir, model_name))
        print(f"  -> 模型已保存: {model_name}")

        metrics = {
            "hsi_nmse": [],
            "accuracy": [],
            "hsi_nmse_std": [],
            "accuracy_std": [],
        }

        for snr_db in snr_values:
            eval_result = evaluate_at_snr(
                model, test_loader, test_x_norm, test_y, device, snr_db, channel_type=channel_type, num_repeats=eval_repeats
            )
            metrics["hsi_nmse"].append(eval_result["hsi_nmse"])
            metrics["accuracy"].append(eval_result["accuracy"])
            metrics["hsi_nmse_std"].append(eval_result["hsi_nmse_std"])
            metrics["accuracy_std"].append(eval_result["accuracy_std"])
            print(
                f"  SNR={snr_db:>3} dB | "
                f"HSI NMSE={eval_result['hsi_nmse']:.6f}±{eval_result['hsi_nmse_std']:.6f} | "
                f"Accuracy={eval_result['accuracy']:.4f}±{eval_result['accuracy_std']:.4f}"
            )

        results_by_dim[total_dim] = metrics
        reconstructions[total_dim] = reconstruct_full_maps(
            model,
            hsi_map,
            device,
            snr_db=train_snr_db,
            channel_type=channel_type,
            hsi_stats=hsi_stats,
        )

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
        valid_mask=valid_mask if show_valid_region_only else None,
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
                "cls_loss_weight": cls_loss_weight,
                "show_valid_region_only": show_valid_region_only,
                "normalization": {
                    "type": "zscore_train_stats",
                    "hsi_mean_shape": list(hsi_stats["mean"].shape),
                    "hsi_std_shape": list(hsi_stats["std"].shape),
                },
                "results": results_by_dim,
            },
            file,
            indent=2,
        )

    print("\n[OK] 模态2实验全部完成。")
    print(f"输出目录: {output_dir}")
    print(f"指标图: {metrics_fig_name}")
    print(f"重构图: {recon_fig_name}")
    print(f"指标JSON: {metrics_json_name}")


if __name__ == "__main__":
    main()