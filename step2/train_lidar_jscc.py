import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from kitti_lidar_dataset import KittiLidarBEVDataset

class LidarJSCC(nn.Module):
    def __init__(self):
        super(LidarJSCC, self).__init__()
        
        # 编码器：输入双通道 BEV [B, 2, 256, 256]
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # -> [B, 16, 128, 128]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # -> [B, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2), # -> [B, 32, 32, 32] 瓶颈特征空间
            nn.ReLU()
        )
        
        # 解码器：还原 3D 点云的 BEV 空间特征
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> [B, 2, 256, 256]
            nn.Tanh() # 高度 Z 有正有负，用 Tanh 容纳更广的范围
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

# ========================================================
# 🚀 训练闭环
# ========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = KittiLidarBEVDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = LidarJSCC().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30 # 吸取刚才图像的经验，直接拉到 30 轮看效果
print("开始执行点云模态单独训练测试...")

for epoch in range(epochs):
    total_loss = 0.0
    for batch_bev in loader:
        batch_bev = batch_bev.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_bev)
        loss = criterion(outputs, batch_bev)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_bev.size(0)
    print(f"Epoch [{epoch+1}/{epochs}] | Lidar BEV Loss: {total_loss / len(dataset):.6f}")

print("[✔] 点云模态重构训练完成！正在对比重构效果...")

# ========================================================
# 📊 结果可视化：展示点云反射强度的重构对比
# ========================================================
model.eval()
with torch.no_grad():
    test_bev = dataset[0].unsqueeze(0).to(device)
    recon_bev = model(test_bev).cpu().squeeze(0).numpy()
    orig_bev = test_bev.cpu().squeeze(0).numpy()

# 提取通道 1 (Intensity 反射强度通道) 来看环境物体的重构清晰度
orig_intensity = orig_bev[1, :, :]
recon_intensity = recon_bev[1, :, :]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(orig_intensity, cmap='viridis')
axes[0].set_title("Original LiDAR BEV (Intensity Channel)", fontsize=10, fontweight="bold")
axes[0].axis('off')

axes[1].imshow(recon_intensity, cmap='viridis')
axes[1].set_title("Reconstructed LiDAR BEV (No Noise Baseline)", fontsize=10, fontweight="bold")
axes[1].axis('off')

plt.tight_layout()
plt.show()