import json
import os
from datetime import datetime

# 解决部分Windows环境下OpenMP运行时重复初始化的Bug
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MultiModalToTextSys(nn.Module):
    """模式三：多模态转文本语义通信网络
    输入 HSI + LiDAR 特征，编码为结构化文本 Token 嵌入，通过物理信道后进行文本解码与分类决策。
    """

    def __init__(self, hsi_dim: int, lidar_dim: int, vocab_size: int, seq_len: int, latent_dim: int, num_classes: int):
        super(MultiModalToTextSys, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # 多模态融合编码器
        self.fusion_encoder = nn.Sequential(
            nn.Linear(hsi_dim + lidar_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, seq_len * latent_dim),  # 生成一串文本符号嵌入
        )
        
        self.latent_dim = latent_dim

        # 接收端：文本符号解码头
        self.text_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, vocab_size)
        )
        
        # 接收端：地物分类决策头
        self.classifier = nn.Linear(seq_len * latent_dim, num_classes)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor, snr_db=None, channel_type="awgn"):
        batch_size = hsi.size(0)
        
        # 跨模态特征拼接融合
        x_fused = torch.cat([hsi, lidar], dim=1)
        semantic_features = self.fusion_encoder(x_fused)
        
        # 重塑为文本 Token 嵌入流形状: [Batch, Seq_Len, Latent_Dim]
        semantic_tokens = semantic_features.view(batch_size, self.seq_len, self.latent_dim)

        # 物理信道扰动模拟
        if snr_db is not None:
            signal_power = torch.mean(semantic_tokens ** 2)
            noise_variance = signal_power / (10 ** (snr_db / 10.0))
            
            if channel_type == "awgn":
                noise = torch.randn_like(semantic_tokens) * torch.sqrt(noise_variance + 1e-12)
                semantic_tokens = semantic_tokens + noise
                
            elif channel_type == "rayleigh":
                h_real = torch.randn(batch_size, 1, 1, device=semantic_tokens.device) / np.sqrt(2)
                h_imag = torch.randn(batch_size, 1, 1, device=semantic_tokens.device) / np.sqrt(2)
                h = torch.sqrt(h_real ** 2 + h_imag ** 2)
                
                noise = torch.randn_like(semantic_tokens) * torch.sqrt(noise_variance + 1e-12)
                received = h * semantic_tokens + noise
                
                # 接收端语义迫零均衡
                semantic_tokens = received / (h + 1e-6)
            else:
                raise ValueError(f"Unsupported channel type: {channel_type}")

        # 接收端并行解码
        # 1. 文本解码
        text_logits = self.text_decoder(semantic_tokens) # [Batch, Seq_Len, Vocab_Size]
        
        # 2. 分类决策
        flattened_semantic = semantic_tokens.view(batch_size, -1)
        class_logits = self.classifier(flattened_semantic)
        
        return text_logits, class_logits


def load_mat_data(mat_path: str):
    """鲁棒读取 .mat 文件，自动过滤系统默认Key"""
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
    """标准化高光谱数据波段"""
    arr = np.asarray(hsi_array)
    if arr.shape[:2] == (166, 600):
        arr = np.transpose(arr, (1, 0, 2))
    return arr.astype(np.float32)


def normalize_lidar_array(lidar_array: np.ndarray) -> np.ndarray:
    """标准化并扩展 LiDAR 数组维度"""
    arr = np.asarray(lidar_array)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[:2] == (166, 600):
        arr = np.transpose(arr, (1, 0, 2))
    return arr.astype(np.float32)


def normalize_gt_array(gt_array: np.ndarray) -> np.ndarray:
    """标准化标签数组维度至 (600, 166)"""
    arr = np.asarray(gt_array)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.shape == (166, 600):
        arr = arr.T
    return arr.astype(np.int32)


def load_italy_multimodal_dataset():
    """同时加载 HSI 和 LiDAR 数据，实现双模态融合对齐"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hsi_path = os.path.join(current_dir, "Italy_hsi.mat")
    lidar_path = os.path.join(current_dir, "Italy_lidar.mat")
    fallback_gt_path = os.path.join(current_dir, "allgrd.mat")

    raw_hsi = load_mat_data(hsi_path)
    raw_lidar = load_mat_data(lidar_path)
    raw_gt = load_mat_data(fallback_gt_path)

    hsi_key = [key for key in raw_hsi.keys() if not key.startswith("_")][0]
    lidar_key = [key for key in raw_lidar.keys() if not key.startswith("_")][0]
    gt_key = [key for key in raw_gt.keys() if not key.startswith("_")][0]

    hsi_map = normalize_hsi_array(raw_hsi[hsi_key])
    lidar_map = normalize_lidar_array(raw_lidar[lidar_key])
    gt_map = normalize_gt_array(raw_gt[gt_key])

    hsi_flat = hsi_map.reshape(-1, hsi_map.shape[2])
    lidar_flat = lidar_map.reshape(-1, lidar_map.shape[2])
    gt_flat = gt_map.reshape(-1)
    
    valid_mask_flat = gt_flat > 0
    valid_hsi = hsi_flat[valid_mask_flat].astype(np.float32)
    valid_lidar = lidar_flat[valid_mask_flat].astype(np.float32)
    valid_labels = (gt_flat[valid_mask_flat] - 1).astype(np.int64)
    num_classes = int(valid_labels.max()) + 1 if valid_labels.size > 0 else 0
    valid_mask = valid_mask_flat.reshape(gt_map.shape[0], gt_map.shape[1])

    return hsi_map, lidar_map, gt_map, valid_mask, valid_hsi, valid_lidar, valid_labels, num_classes


def generate_mock_text_tokens(labels: np.ndarray, vocab_size: int, seq_len: int) -> np.ndarray:
    """【语义通信仿真核心】将各个地物类别翻译映射为离散的高级文本描述 Token 序列。
    例如：类别0映射为带有特定坐标、高度和周边环境关键字的离散 Token 编码向量。
    """
    np.random.seed(42)
    tokens = np.zeros((labels.shape[0], seq_len), dtype=np.int64)
    for idx, label in enumerate(labels):
        # 基于地物标签生成确定性的文本符号流（模拟遥感大模型的文本转译输出）
        base_token_seq = [(int(label) * 3 + i) % (vocab_size - 2) + 1 for i in range(seq_len)]
        tokens[idx] = base_token_seq
    return tokens


def normalize_by_train_stats(train_arr, test_arr):
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((train_arr - mean) / std).astype(np.float32), ((test_arr - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def evaluate_at_snr(model, test_hsi_norm, test_lidar_norm, test_text_tokens, test_labels, device, snr_db, channel_type, num_repeats=5, batch_size=512):
    """评估系统在不同信噪比下的文本翻译错误率（TER）和分类准确率"""
    model.eval()
    ter_list = []
    acc_list = []

    for _ in range(num_repeats):
        total_token_errors = 0
        total_tokens = 0
        correct_cls = 0
        total_samples = 0
        
        with torch.no_grad():
            for start in range(0, len(test_hsi_norm), batch_size):
                end = min(len(test_hsi_norm), start + batch_size)
                b_hsi = torch.from_numpy(test_hsi_norm[start:end]).to(device)
                b_lidar = torch.from_numpy(test_lidar_norm[start:end]).to(device)
                b_tokens = test_text_tokens[start:end]
                b_labels = test_labels[start:end]

                text_logits, class_logits = model(b_hsi, b_lidar, snr_db=snr_db, channel_type=channel_type)
                
                # 计算文本词错率 Token Error Rate (TER)
                pred_tokens = torch.argmax(text_logits, dim=2).cpu().numpy()
                total_token_errors += np.sum(pred_tokens != b_tokens)
                total_tokens += b_tokens.size
                
                # 计算分类准确率
                preds_cls = torch.argmax(class_logits, dim=1).cpu().numpy()
                correct_cls += np.sum(preds_cls == b_labels)
                total_samples += b_labels.shape[0]

        ter_list.append(total_token_errors / max(total_tokens, 1))
        acc_list.append(correct_cls / max(total_samples, 1))

    return {
        "text_ter": float(np.mean(ter_list)),
        "text_ter_std": float(np.std(ter_list)),
        "accuracy": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list)),
    }


def save_metric_curves(results_by_dim, snr_values, output_path, channel_type="awgn"):
    """绘制模式三专属学术指标曲线：左图为文本翻译符号错误率（TER），右图为分类准确率"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for dim, metrics in results_by_dim.items():
        # 左图：高级文本语义恢复错误率 (向下收敛，完美对齐模式二重构曲线趋势)
        axes[0].plot(snr_values, metrics["text_ter"], marker="o", linewidth=2, label=f"Semantic Dim={dim}")
        axes[0].fill_between(snr_values, np.array(metrics["text_ter"]) - np.array(metrics["text_ter_std"]), np.array(metrics["text_ter"]) + np.array(metrics["text_ter_std"]), alpha=0.15)
        
        # 右图：决策层分类准确率
        axes[1].plot(snr_values, metrics["accuracy"], marker="^", linewidth=2, label=f"Semantic Dim={dim}")
        axes[1].fill_between(snr_values, np.array(metrics["accuracy"]) - np.array(metrics["accuracy_std"]), np.array(metrics["accuracy"]) + np.array(metrics["accuracy_std"]), alpha=0.15)

    axes[0].set_title("Text Description Error (TER) vs. SNR", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Received SNR (dB)")
    axes[0].set_ylabel("Token Error Rate (TER)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("Downstream Classification Accuracy vs. SNR", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Received SNR (dB)")
    axes[1].set_ylabel("Classification Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.suptitle(f"Mode 3 ({channel_type.upper()} Channel): Multi-Modal to Text Semantic Translation Performance", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已成功将学术指标性能曲线保存至: {output_path}")


def reconstruct_full_text_maps(model, hsi_map, lidar_map, hsi_stats, lidar_stats, test_tokens_full, device, snr_db, channel_type, batch_size=512):
    """对大幅全景地图进行全量滑窗推断，计算每个像素位置处的文本描述生成正确率"""
    height, width, _ = hsi_map.shape
    flat_hsi = hsi_map.reshape(-1, hsi_map.shape[2]).astype(np.float32)
    flat_lidar = lidar_map.reshape(-1, lidar_map.shape[2]).astype(np.float32)
    
    norm_hsi = (flat_hsi - hsi_stats["mean"]) / hsi_stats["std"]
    norm_lidar = (flat_lidar - lidar_stats["mean"]) / lidar_stats["std"]

    correctness_list = []
    model.eval()
    with torch.no_grad():
        for start in range(0, norm_hsi.shape[0], batch_size):
            end = min(norm_hsi.shape[0], start + batch_size)
            b_hsi = torch.from_numpy(norm_hsi[start:end]).to(device)
            b_lidar = torch.from_numpy(norm_lidar[start:end]).to(device)
            b_tokens = test_tokens_full[start:end]

            text_logits, _ = model(b_hsi, b_lidar, snr_db=snr_db, channel_type=channel_type)
            pred_tokens = torch.argmax(text_logits, dim=2).cpu().numpy()
            
            # 评估每个像素转换出的文本短语正确比例 (0.0 ~ 1.0)
            pixel_accuracy = np.mean(pred_tokens == b_tokens, axis=1)
            correctness_list.append(pixel_accuracy)

    return np.concatenate(correctness_list).reshape(height, width)


def save_reconstruction_figure(hsi_map, text_maps, visual_dims, output_path, snr_db, channel_type, valid_mask):
    """空间重构可视化：将文本翻译结果可视化为全图“语义文本可解码度/正确率映射图”，彰显低带宽文本传输威力"""
    band_idx = 9
    hsi_vmin, hsi_vmax = float(hsi_map[:, :, band_idx].min()), float(hsi_map[:, :, band_idx].max())
    original_hsi = np.where(valid_mask, hsi_map[:, :, band_idx], np.nan)

    ncols = 1 + len(visual_dims)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))

    axes[0].imshow(original_hsi, cmap="jet", vmin=hsi_vmin, vmax=hsi_vmax)
    axes[0].set_title(f"Original HSI Source\n(Band {band_idx+1})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for idx, dim in enumerate(visual_dims):
        t_map = text_maps[dim].copy()
        t_map = np.where(valid_mask, t_map, np.nan)
        im = axes[idx + 1].imshow(t_map, cmap="plasma", vmin=0.0, vmax=1.0)
        axes[idx + 1].set_title(f"Decoded Text Accuracy Map\n(Semantic Dim = {dim})", fontsize=11)
        axes[idx + 1].axis("off")

    plt.suptitle(f"Mode 3 Multi-Modal to Text Correctness Mapping Under {channel_type.upper()} Channel (SNR={snr_db} dB)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已成功将高级文本描述生成正确率图保存至: {output_path}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hsi_map, lidar_map, gt_map, valid_mask, valid_hsi, valid_lidar, valid_labels, num_classes = load_italy_multimodal_dataset()

    # 离散语义文本符号参数（模拟翻译词表）
    vocab_size = 50 
    seq_len = 8      # 每组多模态像素凝练为 8 个文本 Token (如：坐标、类型、高度、环境)
    
    # 核心超参数配置（与模式二完全对齐控制变量）
    batch_size = 128
    epochs = 20
    learning_rate = 0.002
    train_snr_db = 10
    channel_type = "awgn"  # 高斯白噪声信道
    snr_values = [-5, 0, 5, 10, 15, 20]
    visual_dims = [2, 6, 12]   # 文本流中每个符号嵌入的隐层维度特征瓶颈
    eval_repeats = 5
    cls_loss_weight = 0.5

    # 划分训练/测试集
    indices = np.arange(valid_hsi.shape[0])
    tr_idx, te_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=valid_labels)
    
    # 标准化
    tr_hsi_norm, te_hsi_norm, hsi_stats = normalize_by_train_stats(valid_hsi[tr_idx], valid_hsi[te_idx])
    tr_lidar_norm, te_lidar_norm, lidar_stats = normalize_by_train_stats(valid_lidar[tr_idx], valid_lidar[te_idx])
    
    # 生成对应的文本 Token 标签
    valid_text_tokens = generate_mock_text_tokens(valid_labels, vocab_size, seq_len)
    tr_tokens, te_tokens = valid_text_tokens[tr_idx], valid_text_tokens[te_idx]

    train_dataset = TensorDataset(torch.from_numpy(tr_hsi_norm), torch.from_numpy(tr_lidar_norm), torch.from_numpy(tr_tokens), torch.from_numpy(valid_labels[tr_idx]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("====================================================================")
    print(" 正在启动优化版『模式三：多模态转文本极低带宽语义通信网络』分支实验")
    print("====================================================================")
    print(f"运行设备: {device} | 文本序列长度: {seq_len} Tokens | 衰落信道: {channel_type}")

    results_by_dim = {}
    text_maps = {}

    for latent_dim in visual_dims:
        print(f"\n[阶梯训练] 正在训练跨模态图文转译网络，符号嵌入特征维度 = {latent_dim}...")
        model = MultiModalToTextSys(hsi_dim=hsi_map.shape[2], lidar_dim=lidar_map.shape[2], vocab_size=vocab_size, seq_len=seq_len, latent_dim=latent_dim, num_classes=num_classes).to(device)
        
        text_criterion = nn.CrossEntropyLoss()
        cls_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 多任务联合训练：兼顾文本精准转译与下游决策
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for b_hsi, b_lidar, b_tok, b_lab in train_loader:
                b_hsi, b_lidar, b_tok, b_lab = b_hsi.to(device), b_lidar.to(device), b_tok.to(device), b_lab.to(device)
                optimizer.zero_grad()
                
                text_logits, class_logits = model(b_hsi, b_lidar, snr_db=train_snr_db, channel_type=channel_type)
                
                loss_text = text_criterion(text_logits.view(-1, vocab_size), b_tok.view(-1))
                loss_cls = cls_criterion(class_logits, b_lab)
                
                loss = loss_text + cls_loss_weight * loss_cls
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch + 1}/{epochs}] | 联合语义损失 (Joint Loss): {epoch_loss / len(train_loader):.6f}")

        # 稳健性测试
        metrics = {"text_ter": [], "text_ter_std": [], "accuracy": [], "accuracy_std": []}
        for snr_db in snr_values:
            eval_result = evaluate_at_snr(model, te_hsi_norm, te_lidar_norm, te_tokens, valid_labels[te_idx], device, snr_db, channel_type, num_repeats=eval_repeats)
            metrics["text_ter"].append(eval_result["text_ter"])
            metrics["text_ter_std"].append(eval_result["text_ter_std"])
            metrics["accuracy"].append(eval_result["accuracy"])
            metrics["accuracy_std"].append(eval_result["accuracy_std"])
            print(f"    SNR={snr_db:>3} dB | 文本词错率 TER={eval_result['text_ter']:.4f} | 分类准确率 Accuracy={eval_result['accuracy']:.4f}")

        results_by_dim[latent_dim] = metrics
        
        # 全图大图的离散文本可解译度制图
        full_gt_flat = gt_map.reshape(-1)
        full_mock_tokens = generate_mock_text_tokens(full_gt_flat, vocab_size, seq_len)
        text_maps[latent_dim] = reconstruct_full_text_maps(model, hsi_map, lidar_map, hsi_stats, lidar_stats, full_mock_tokens, device, snr_db=train_snr_db, channel_type=channel_type)

        model_save_name = f"mode3_model_dim{latent_dim}_{channel_type}_{timestamp}.pth"
        model_save_path = os.path.join(output_dir, model_save_name)
        
        # 保存模型权重以及标准化的统计量（写论文或后续推理时必须用配套的 stats 逆标准化）
        torch.save({
            'model_state_dict': model.state_dict(),
            'latent_dim': latent_dim,
            'vocab_size': vocab_size,
            'seq_len': seq_len,
            'hsi_stats': hsi_stats,
            'lidar_stats': lidar_stats
        }, model_save_path)
        print(f"    [💾] 已将该维度模型及统计量完整保存至: {model_save_path}")

    # 导出核心图表
    metrics_fig_name = f"mode3_text_metrics_curves_{channel_type}_{timestamp}.png"
    recon_fig_name = f"mode3_text_correctness_mapping_{channel_type}_{timestamp}.png"

    save_metric_curves(results_by_dim, snr_values, os.path.join(output_dir, metrics_fig_name), channel_type=channel_type)
    save_reconstruction_figure(hsi_map, text_maps, visual_dims, os.path.join(output_dir, recon_fig_name), snr_db=train_snr_db, channel_type=channel_type, valid_mask=valid_mask)

    print("\n[✔] 模式三分支（图文语义通信）实验顺利闭环！所有高清学术图表已保存至 outputs 目录。")


if __name__ == "__main__":
    main()