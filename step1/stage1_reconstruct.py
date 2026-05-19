import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ========================================================
# 1. 动态输入维度的语义自编码器
# ========================================================
class SemanticAutoEncoder(nn.Module):
    def __init__(self, input_dim, semantic_dim=16):
        """
        input_dim: 自动读取的真实通道数 (比如 48)
        semantic_dim: 压缩后的语义维度
        """
        super(SemanticAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, semantic_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(semantic_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        s = self.encoder(x)
        x_hat = self.decoder(s)
        return x_hat, s

# ========================================================
# 2. 数据加载 (自动推断维度)
# ========================================================
def load_imaging_data():
    print("正在加载数据集...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hsi_path = os.path.join(current_dir, 'Houston13.mat')
    gt_path = os.path.join(current_dir, 'Houston13_7gt.mat')
    
    with h5py.File(hsi_path, 'r') as f:
        data = np.array(f['ori_data']).transpose(2, 1, 0)
    with h5py.File(gt_path, 'r') as f:
        gt = np.array(f['map']).transpose(1, 0)
        
    H, W, C = data.shape  # C 可能是 48，也可能是 145，全自动读取
    
    data_flat = data.reshape(-1, C)
    gt_flat = gt.reshape(-1)
    
    valid_idx = np.where(gt_flat > 0)[0]
    X_valid = data_flat[valid_idx]
    
    print(f"数据加载成功！地图尺寸: {H}x{W}，真实通道数: {C}，有效像素点: {len(X_valid)}")
    dataset = TensorDataset(torch.from_numpy(X_valid).float())
    return dataset, H, W, C, data, gt

# ========================================================
# 3. 主流程与动态可视化
# ========================================================
def main():
    SEMANTIC_DIM = 8
    BATCH_SIZE = 128
    EPOCHS = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
    
    dataset, H, W, C, original_map, gt_map = load_imaging_data()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 【动态实例化】把真实的通道数 C 传给模型
    model = SemanticAutoEncoder(input_dim=C, semantic_dim=SEMANTIC_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 3.1 训练
    print(f"\n--- 开始训练 (输入维度: {C} -> 语义维度: {SEMANTIC_DIM}) ---")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            batch_x_hat, _ = model(batch_x)
            loss = criterion(batch_x_hat, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] | MSE Loss: {epoch_loss/len(loader):.6f}")
        
    # 3.2 分批重构
    print("\n正在生成重构图...")
    model.eval()
    X_all_np = original_map.reshape(-1, C)
    reconstructed_flat = []
    
    EVAL_BATCH_SIZE = 512
    with torch.no_grad():
        for i in range(0, len(X_all_np), EVAL_BATCH_SIZE):
            chunk = X_all_np[i:i+EVAL_BATCH_SIZE]
            chunk_tensor = torch.from_numpy(chunk).float().to(device)
            
            if chunk_tensor.shape[0] <= 1:
                chunk_hat = chunk_tensor
            else:
                chunk_hat, _ = model(chunk_tensor)
                
            reconstructed_flat.append(chunk_hat.cpu().numpy())
            
    reconstructed_map = np.vstack(reconstructed_flat).reshape(H, W, C)
    
    # 3.3 安全绘图 (绝不越界)
    # 动态挑选两个波段：比如第 5 波段和 接近末尾的波段
    band_1 = min(5, C - 1)
    band_2 = min(30, C - 1)
    
    plt.figure(figsize=(12, 6))
    
    # 波段 1 对比
    plt.subplot(2, 2, 1)
    plt.imshow(original_map[:, :, band_1], cmap='jet')
    plt.title(f"Original HSI (Band {band_1})")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(reconstructed_map[:, :, band_1], cmap='jet')
    plt.title(f"Reconstructed HSI (Band {band_1})")
    plt.axis('off')
    
    # 波段 2 对比
    plt.subplot(2, 2, 3)
    plt.imshow(original_map[:, :, band_2], cmap='viridis')
    plt.title(f"Original HSI (Band {band_2})")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(reconstructed_map[:, :, band_2], cmap='viridis')
    plt.title(f"Reconstructed HSI (Band {band_2})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('semantic_reconstruction_result.png', dpi=300)
    print("\n[✔] 成功！重构对比效果图已生成。")

if __name__ == "__main__":
    main()