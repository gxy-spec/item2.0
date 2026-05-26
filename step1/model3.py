import json
import os
import math
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

# ====================================================================
# 【显式文本翻译系统】定义结构化的遥感语义符号映射词表 (大小固定为50)
# ====================================================================
# 索引分配策略：
# 0: PAD, 1: UNK
# 2-9: 地物类型 (Type) | 10-19: 高程特征 (Elevation)
# 20-29: 表面材质 (Surface) | 30-49: 环境/安全预警 (Alert Status)
VOCAB_WORDS = ["PAD", "UNK"] * 25  # 占位初始化
VOCAB_MAPPING = {
    # 类属引导符
    0: "PAD", 1: "UNK",
    # 1. 地物类型 (Tokens 2-9)
    2: "Residential Zone", 3: "Industrial Park", 4: "Agricultural Field", 
    5: "Airport Runway", 6: "River & Waterway", 7: "Commercial Area",
    # 2. 高程特征 (Tokens 10-19)
    10: "Height: Low (<5m)", 11: "Height: Medium (5-12m)", 12: "Height: High (12-35m)", 
    13: "Height: Ultra-High (>35m)",
    # 3. 表面材质 (Tokens 20-29)
    20: "Asphalt Surface", 21: "Bare Soil & Mud", 22: "Dense Vegetation", 
    23: "Reinforced Concrete", 24: "Clear Water Body",
    # 4. 环境与安全预警 (Tokens 30-49)
    30: "[Status: Safe Environment]", 31: "[Status: Flood Risk Alert]", 
    32: "[Status: Thermal Anomaly Detected]", 33: "[Status: Normal Operational]"
}

# 将映射刷入全局词表数组中以供索引查询
for idx, word in VOCAB_MAPPING.items():
    if idx < 50:
        VOCAB_WORDS[idx] = word


class MultiModalToTextSys(nn.Module):
    """支持显式语义翻译的多模态转文本语义通信网络"""
    def __init__(self, hsi_dim: int, lidar_dim: int, vocab_size: int, seq_len: int, latent_dim: int, num_classes: int):
        super(MultiModalToTextSys, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # 方案 B：非对等预特征提取流
        self.hsi_stem = nn.Sequential(
            nn.Linear(hsi_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True)
        )
        self.lidar_stem = nn.Sequential(
            nn.Linear(lidar_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        # 跨模态融合编码器
        self.fusion_encoder = nn.Sequential(
            nn.Linear(96 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, seq_len * latent_dim), 
        )
        
        # 接收端：文本符号解码头
        self.text_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, vocab_size)
        )
        
        # 方案 A：升级为非线性地物分类决策头
        self.classifier = nn.Sequential(
            nn.Linear(seq_len * latent_dim, seq_len * latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(seq_len * latent_dim * 2),
            nn.Linear(seq_len * latent_dim * 2, num_classes)
        )

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor, snr_db=None, channel_type="awgn"):
        batch_size = hsi.size(0)
        
        feat_hsi = self.hsi_stem(hsi)
        feat_lidar = self.lidar_stem(lidar)
        x_fused = torch.cat([feat_hsi, feat_lidar], dim=1)
        
        semantic_features = self.fusion_encoder(x_fused)
        semantic_tokens = semantic_features.view(batch_size, self.seq_len, self.latent_dim)

        if snr_db is not None:
            signal_power = torch.mean(semantic_tokens ** 2)
            noise_variance = signal_power / (10 ** (snr_db / 10.0))
            
            if channel_type == "awgn":
                noise = torch.randn_like(semantic_tokens) * torch.sqrt(noise_variance + 1e-12)
                semantic_tokens = semantic_tokens + noise
                
            elif channel_type == "rayleigh":
                h_real = torch.randn(batch_size, 1, 1, device=semantic_tokens.device) / math.sqrt(2)
                h_imag = torch.randn(batch_size, 1, 1, device=semantic_tokens.device) / math.sqrt(2)
                h_mag = torch.sqrt(h_real ** 2 + h_imag ** 2)
                
                noise = torch.randn_like(semantic_tokens) * torch.sqrt(noise_variance + 1e-12)
                received = h_mag * semantic_tokens + noise
                
                epsilon = 1e-2
                semantic_tokens = received * h_mag / (h_mag ** 2 + epsilon)
            else:
                raise ValueError(f"Unsupported channel type: {channel_type}")

        text_logits = self.text_decoder(semantic_tokens)  
        flattened_semantic = semantic_tokens.view(batch_size, -1)
        class_logits = self.classifier(flattened_semantic)
        
        return text_logits, class_logits


def load_mat_data(mat_path: str):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    try:
        mat_contents = sio.loadmat(mat_path)
        keys = [key for key in mat_contents.keys() if not key.startswith("_")]
        return {key: mat_contents[key] for key in keys}
    except Exception as exc:
        raise RuntimeError(f"Failed to load MAT file {mat_path}: {exc}")


def normalize_hsi_array(hsi_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(hsi_array)
    if arr.shape[:2] == (166, 600):
        arr = np.transpose(arr, (1, 0, 2))
    return arr.astype(np.float32)


def normalize_lidar_array(lidar_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(lidar_array)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[:2] == (166, 600):
        arr = np.transpose(arr, (1, 0, 2))
    return arr.astype(np.float32)


def normalize_gt_array(gt_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(gt_array)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.shape == (166, 600):
        arr = arr.T
    return arr.astype(np.int32)


def load_italy_multimodal_dataset():
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


def generate_explicit_text_tokens(labels: np.ndarray, seq_len: int) -> np.ndarray:
    """【显式符号转换】将地物真实标签转换为强物理对应的结构化 Token 序列"""
    np.random.seed(42)
    tokens = np.zeros((labels.shape[0], seq_len), dtype=np.int64)
    
    for idx, label in enumerate(labels):
        # 根据 Trento 数据集类别先验赋予极具解释性的属性组合
        if label == 0:    # 建筑物 / 住宅区
            seq = [2, 11, 23, 33] 
        elif label == 1:  # 果树 / 农业区
            seq = [4, 10, 22, 30]
        elif label == 2:  # 道路 / 沥青
            seq = [7, 10, 20, 33]
        elif label == 3:  # 设施 / 工业区
            seq = [3, 12, 23, 32]
        elif label == 4:  # 葡萄园
            seq = [4, 11, 22, 30]
        else:             # 默认通用遥感属性
            seq = [2, 10, 21, 33]
            
        # 用 PAD 补齐多余的序列长度
        while len(seq) < seq_len:
            seq.append(0)
        tokens[idx] = seq[:seq_len]
    return tokens


def decode_tokens_to_string(tokens: np.ndarray) -> str:
    """将预测的独立离散符号 Token 流级联翻译成一句可读性极强的属性短语"""
    phrases = []
    for t in tokens:
        t_int = int(t)
        if t_int in [0, 1]: # 跳过不展示填充符
            continue
        if t_int in VOCAB_MAPPING:
            phrases.append(VOCAB_MAPPING[t_int])
        else:
            phrases.append(f"Token_{t_int}")
            
    if not phrases:
        return "Empty Semantic Stream (Noise Overblown)"
    return " -> ".join(phrases)


def normalize_by_train_stats(train_arr, test_arr):
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((train_arr - mean) / std).astype(np.float32), ((test_arr - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def evaluate_at_snr(model, test_hsi_norm, test_lidar_norm, test_text_tokens, test_labels, device, snr_db, channel_type, num_repeats=5, batch_size=512):
    model.eval()
    similarity_list = []
    acc_list = []

    with torch.no_grad():
        for _ in range(num_repeats):
            total_correct_tokens = 0
            total_tokens = 0
            correct_cls = 0
            total_samples = 0
            
            for start in range(0, len(test_hsi_norm), batch_size):
                end = min(len(test_hsi_norm), start + batch_size)
                b_hsi = torch.from_numpy(test_hsi_norm[start:end]).to(device)
                b_lidar = torch.from_numpy(test_lidar_norm[start:end]).to(device)
                b_tokens = test_text_tokens[start:end]
                b_labels = test_labels[start:end]

                text_logits, class_logits = model(b_hsi, b_lidar, snr_db=snr_db, channel_type=channel_type)
                
                pred_tokens = torch.argmax(text_logits, dim=2).cpu().numpy()
                total_correct_tokens += np.sum(pred_tokens == b_tokens)
                total_tokens += b_tokens.size
                
                preds_cls = torch.argmax(class_logits, dim=1).cpu().numpy()
                correct_cls += np.sum(preds_cls == b_labels)
                total_samples += b_labels.shape[0]

            similarity_list.append(total_correct_tokens / max(total_tokens, 1))
            acc_list.append(correct_cls / max(total_samples, 1))

    return {
        "text_sim": float(np.mean(similarity_list)),
        "text_sim_std": float(np.std(similarity_list)),
        "accuracy": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list)),
    }


def save_metric_curves(results_by_dim, snr_values, output_path, channel_type="awgn"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for dim, metrics in results_by_dim.items():
        axes[0].plot(snr_values, metrics["text_sim"], marker="o", linewidth=2, label=f"Semantic Dim={dim}")
        axes[0].fill_between(snr_values, np.array(metrics["text_sim"]) - np.array(metrics["text_sim_std"]), np.array(metrics["text_sim"]) + np.array(metrics["text_sim_std"]), alpha=0.15)
        
        axes[1].plot(snr_values, metrics["accuracy"], marker="^", linewidth=2, label=f"Semantic Dim={dim}")
        axes[1].fill_between(snr_values, np.array(metrics["accuracy"]) - np.array(metrics["accuracy_std"]), np.array(metrics["accuracy"]) + np.array(metrics["accuracy_std"]), alpha=0.15)

    axes[0].set_title("Text Semantic Similarity vs. SNR", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Received SNR (dB)")
    axes[0].set_ylabel("Text Matching Similarity")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("Downstream Classification Accuracy vs. SNR", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Received SNR (dB)")
    axes[1].set_ylabel("Classification Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.suptitle(f"Mode 3 ({channel_type.upper()} Channel): Multi-Modal Explicit Semantic Curves", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def reconstruct_full_text_maps(model, hsi_map, lidar_map, hsi_stats, lidar_stats, test_tokens_full, device, snr_db, channel_type, batch_size=512):
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
            
            pixel_accuracy = np.mean(pred_tokens == b_tokens, axis=1)
            correctness_list.append(pixel_accuracy)

    return np.concatenate(correctness_list).reshape(height, width)


def save_reconstruction_figure(hsi_map, text_maps, visual_dims, output_path, snr_db, channel_type, valid_mask):
    band_idx = 9
    hsi_vmin, hsi_vmax = float(hsi_map[:, :, band_idx].min()), float(hsi_map[:, :, band_idx].max())
    original_hsi = hsi_map[:, :, band_idx].copy()

    ncols = 1 + len(visual_dims)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))

    axes[0].imshow(original_hsi, cmap="jet", vmin=hsi_vmin, vmax=hsi_vmax)
    axes[0].set_title(f"Original HSI Source\n(Band {band_idx+1})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for idx, dim in enumerate(visual_dims):
        t_map = text_maps[dim].copy()
        t_map[~valid_mask] = 0.0  
        axes[idx + 1].imshow(t_map, cmap="plasma", vmin=0.0, vmax=1.0)
        axes[idx + 1].set_title(f"Explicit Semantic Map\n(Semantic Dim = {dim})", fontsize=11)
        axes[idx + 1].axis("off")

    plt.suptitle(f"Mode 3 Translation Accuracy Mapping under {channel_type.upper()} Channel", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hsi_map, lidar_map, gt_map, valid_mask, valid_hsi, valid_lidar, valid_labels, num_classes = load_italy_multimodal_dataset()

    vocab_size = 50 
    seq_len = 6  # 缩短序列长度更契合四元属性组合表达提升密度
    
    batch_size = 512
    epochs = 30
    learning_rate = 0.005
    train_snr_db = 10
    channel_type = "awgn"  
    snr_values = [-5, 0, 5, 10, 15, 20]
    visual_dims = [2, 6, 12]   
    eval_repeats = 5
    cls_loss_weight = 0.5      

    indices = np.arange(valid_hsi.shape[0])
    tr_idx, te_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=valid_labels)
    
    tr_hsi_norm, te_hsi_norm, hsi_stats = normalize_by_train_stats(valid_hsi[tr_idx], valid_hsi[te_idx])
    tr_lidar_norm, te_lidar_norm, lidar_stats = normalize_by_train_stats(valid_lidar[tr_idx], valid_lidar[te_idx])
    
    # 转换为显式有物理含义的 Token 序列
    valid_text_tokens = generate_explicit_text_tokens(valid_labels, seq_len)
    tr_tokens, te_tokens = valid_text_tokens[tr_idx], valid_text_tokens[te_idx]

    train_dataset = TensorDataset(torch.from_numpy(tr_hsi_norm), torch.from_numpy(tr_lidar_norm), torch.from_numpy(tr_tokens), torch.from_numpy(valid_labels[tr_idx]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("====================================================================")
    print(" 正在启动优化方案版『模式三：多模态转结构化显式文本语义通信网络』")
    print("====================================================================")

    results_by_dim = {}
    text_maps = {}
    saved_models_dict = {}

    for latent_dim in visual_dims:
        print(f"\n[阶梯训练] 符号嵌入特征维度 = {latent_dim}...")
        model = MultiModalToTextSys(hsi_dim=hsi_map.shape[2], lidar_dim=lidar_map.shape[2], vocab_size=vocab_size, seq_len=seq_len, latent_dim=latent_dim, num_classes=num_classes).to(device)
        
        text_criterion = nn.CrossEntropyLoss()
        cls_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
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

            print(f"  Epoch [{epoch+1}/{epochs}] | Joint Loss: {epoch_loss / len(train_loader):.6f}")

        # 测试验证
        metrics = {"text_sim": [], "text_sim_std": [], "accuracy": [], "accuracy_std": []}
        for snr_db in snr_values:
            eval_result = evaluate_at_snr(model, te_hsi_norm, te_lidar_norm, te_tokens, valid_labels[te_idx], device, snr_db, channel_type, num_repeats=eval_repeats)
            metrics["text_sim"].append(eval_result["text_sim"])
            metrics["text_sim_std"].append(eval_result["text_sim_std"])
            metrics["accuracy"].append(eval_result["accuracy"])
            metrics["accuracy_std"].append(eval_result["accuracy_std"])

        results_by_dim[latent_dim] = metrics
        saved_models_dict[latent_dim] = model
        
        full_gt_flat = gt_map.reshape(-1)
        full_mock_tokens = generate_explicit_text_tokens(full_gt_flat, seq_len)
        text_maps[latent_dim] = reconstruct_full_text_maps(model, hsi_map, lidar_map, hsi_stats, lidar_stats, full_mock_tokens, device, snr_db=train_snr_db, channel_type=channel_type)

    metrics_fig_name = f"mode3_explicit_curves_{channel_type}_{timestamp}.png"
    recon_fig_name = f"mode3_explicit_mapping_{channel_type}_{timestamp}.png"

    save_metric_curves(results_by_dim, snr_values, os.path.join(output_dir, metrics_fig_name), channel_type=channel_type)
    save_reconstruction_figure(hsi_map, text_maps, visual_dims, os.path.join(output_dir, recon_fig_name), snr_db=train_snr_db, channel_type=channel_type, valid_mask=valid_mask)

    # ====================================================================
    # 【显式文本对照生成】抽取典型点生成全场景无线多径衰落翻译明文对照表
    # ====================================================================
    print("\n[📊 语义通信Case Study] 正在导出地物属性在恶劣/中等/无损瑞利信道下的显式翻译对照表...")
    table_save_path = os.path.join(output_dir, "mode3_text_alignment_table.txt")
    
    # 精准抽取代表性遥感地物案例
    case_indices = []
    target_classes = [0, 1, 3]  # 抽取建筑物、农业区、工业区这三类极具属性对比性的样本
    for c in target_classes:
        match_idx = np.where(valid_labels[te_idx] == c)[0]
        if len(match_idx) > 0:
            case_indices.append(match_idx[0])

    eval_snrs = [-5, 5, 20]
    best_model = saved_models_dict[12]  # 选用具有最佳转译带宽的 Dim=12 模型展示语义恢复
    best_model.eval()

    with open(table_save_path, "w", encoding="utf-8") as f:
        f.write("="*115 + "\n")
        f.write(" 模式三 (Mode 3) 遥感多模态联合语义通信网络：典型地物无线瑞利深衰落信道『明文属性转译对照表』\n")
        f.write("="*115 + "\n\n")
        f.write(f"实验配置：信道类型 = {channel_type.upper()} 多径衰落 | 语义瓶颈层特征带宽 = 12 维\n")
        f.write(f"结构化语义解码规范：[地物类型] -> [高程特征] -> [表面材质] -> [环境与安全预警]\n")
        f.write("-" * 115 + "\n\n")
        
        for idx in case_indices:
            hsi_p = te_hsi_norm[idx:idx+1]
            lidar_p = te_lidar_norm[idx:idx+1]
            gt_tok = te_tokens[idx]
            label_p = valid_labels[te_idx][idx]
            
            # 建立可读标签解释
            class_names = {0: "Buildings/Residential", 1: "Agricultural/Trees", 3: "Industrial Facilities"}
            c_name = class_names.get(label_p, f"Class {label_p}")
            
            f.write(f"▶ [地物样本点] 测试集像素索引: #{idx} | 真实核心归属类别: {c_name}\n")
            f.write(f"  ● 发射端输入多模态源文本 (Source Explicit Text):\n")
            f.write(f"    \"{decode_tokens_to_string(gt_tok)}\"\n\n")
            
            for snr in eval_snrs:
                with torch.no_grad():
                    t_logits, _ = best_model(torch.from_numpy(hsi_p).to(device), torch.from_numpy(lidar_p).to(device), snr_db=snr, channel_type=channel_type)
                    p_tok = torch.argmax(t_logits, dim=2).cpu().numpy()[0]
                
                snr_tag = { -5: "【恶劣信道】", 5: "【中等信道】", 20: "【无损信道】" }.get(snr, "")
                f.write(f"  └─ 接收端语义译码明文 (SNR = {snr:>3} dB) {snr_tag}:\n")
                f.write(f"    \"{decode_tokens_to_string(p_tok)}\"\n")
            f.write("\n" + "-"*115 + "\n\n")
            
    print(f"   [✔] 结构化明文属性对照表生成成功，已完美保存至: {table_save_path}")
    print("\n[🎉 完结] 包含显式短语翻译的多模态语义通信系统重构完毕！")


if __name__ == "__main__":
    main()