import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_calc

# 导入多模态数据集
from kitti_multimodal_dataset import KittiMultiModalDataset

# ========================================================
# 1. 📡 稳定版 A2G 复合无线信道层 (带梯度保护)
# ========================================================
class A2GComplexChannel(nn.Module):
    def __init__(self, k_factor=3.0):
        super(A2GComplexChannel, self).__init__()
        self.k_factor = k_factor 

    def forward(self, x, snr_db):
        # 1. 能量归一化 (加上 1e-4 彻底防止分母为 0)
        rms = torch.sqrt(torch.mean(x ** 2) + 1e-4)
        x_normalized = x / rms
        
        # 2. 莱斯小尺度衰落 H 矩阵
        los_weight = np.sqrt(self.k_factor / (self.k_factor + 1))
        nlos_weight = np.sqrt(1 / (self.k_factor + 1))
        H = torch.ones_like(x_normalized) * los_weight + torch.randn_like(x_normalized) * nlos_weight
        
        # 3. 计算噪声功率
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = np.sqrt(1.0 / snr_linear)
        noise = torch.randn_like(x_normalized) * noise_std
        
        # 信号到达接收端
        y_received = H * x_normalized + noise
        
        # 4. 🎯【核心防爆升级】：采用 MMSE (最小均方误差) 均衡原理
        # 哪怕 H 衰落到了 0，分母上依然有噪声功率项 (1/snr_linear) 护航，绝对不可能除以 0！
        # 这是通信仿真里最标准、最优雅的防 nan 做法
        y_equalized = (y_received * H) / (H ** 2 + (1.0 / snr_linear) + 1e-4)
        
        return y_equalized

# ========================================================
# 2. 🧬 加固型多模态端到端语义网络
# ========================================================
class MultiModalJSCCFinal(nn.Module):
    def __init__(self):
        super(MultiModalJSCCFinal, self).__init__()
        self.channel = A2GComplexChannel()
        
        # 图像分支：引入 InstanceNorm2d 强行稳定抗噪特征
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), # 提升通道到32
            nn.InstanceNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 提升通道到64
            nn.InstanceNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
        )
        self.img_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid() 
        )
        
        # 雷达分支
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
        )
        self.lidar_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh() 
        )

    def forward(self, img_x, lidar_x, snr_db):
        img_feat = self.img_encoder(img_x)
        lidar_feat = self.lidar_encoder(lidar_x)
        
        img_feat_ch = self.channel(img_feat, snr_db)
        lidar_feat_ch = self.channel(lidar_feat, snr_db)
        
        return self.img_decoder(img_feat_ch), self.lidar_decoder(lidar_feat_ch)

# ========================================================
# 🚀 训练驱动控制
# ========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_dataset = KittiMultiModalDataset()
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = MultiModalJSCCFinal().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003) # 降低学习率，小步稳健试探信道

# 🚨 请注意：为了让你看到效果，我这里保持 30 轮，如果想要达到完美的学术质量，建议后续改为 100 轮
epochs = 30 
print("正在开启加固后的多模态端到端训练...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_img, batch_lidar in train_loader:
        batch_img, batch_lidar = batch_img.to(device), batch_lidar.to(device)
        
        # 信道热身机制：前几轮不引入太极端的噪声，防止开局死锁
        if epoch < 5:
            random_snr = np.random.uniform(10.0, 20.0)
        else:
            random_snr = np.random.uniform(0.0, 20.0)
            
        optimizer.zero_grad()
        recon_img, recon_lidar = model(batch_img, batch_lidar, random_snr)
        
        loss = criterion(recon_img, batch_img) + 2.0 * criterion(recon_lidar, batch_lidar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_img.size(0)
        
    print(f"Epoch [{epoch+1}/{epochs}] | Train Joint Loss: {total_loss / len(train_dataset):.6f}")

torch.save(model.state_dict(), "best_multimodal_jscc.pth")

# ========================================================
# 📊 测试扫频：生成你梦寐以求的学术曲线数据
# ========================================================
model.eval()
snr_test_list = [-5, 0, 5, 10, 15, 20]
img_psnr_results = []
lidar_nmse_results = []

print("\n开始扫描不同信噪比下的学术性能指标...")
with torch.no_grad():
    for test_snr in snr_test_list:
        current_psnr, current_nmse = [], []
        for t_img, t_lidar in test_loader:
            t_img, t_lidar = t_img.to(device), t_lidar.to(device)
            r_img, r_lidar = model(t_img, t_lidar, test_snr)
            
            img_org = np.clip(t_img.cpu().squeeze(0).numpy().transpose(1, 2, 0), 0, 1)
            img_rec = np.clip(r_img.cpu().squeeze(0).numpy().transpose(1, 2, 0), 0, 1)
            lidar_org = t_lidar.cpu().squeeze(0).numpy()
            lidar_rec = r_lidar.cpu().squeeze(0).numpy()
            
            current_psnr.append(psnr_calc(img_org, img_rec, data_range=1.0))
            current_nmse.append(np.mean((lidar_org - lidar_rec) ** 2) / (np.mean(lidar_org ** 2) + 1e-8))
            
        img_psnr_results.append(np.mean(current_psnr))
        lidar_nmse_results.append(np.mean(current_nmse))

# ========================================================
# 🎨 绘图系统：一网打尽，同时保存并显示两张大图
# ========================================================
# 图一：学术曲线图
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(snr_test_list, img_psnr_results, 'ro-', linewidth=2)
ax1.set_xlabel('Channel SNR (dB)')
ax1.set_ylabel('Image PSNR (dB)')
ax1.set_title('Image Quality vs SNR (Generalization Test)')
ax1.grid(True)

ax2.plot(snr_test_list, lidar_nmse_results, 'g^-', linewidth=2)
ax2.set_xlabel('Channel SNR (dB)')
ax2.set_ylabel('LiDAR BEV NMSE')
ax2.set_title('LiDAR Error vs SNR (Generalization Test)')
ax2.grid(True)
plt.tight_layout()
plt.savefig("academic_curves.png") # 自动保存折线图到本地
print("[✔] 折线图已保存为 academic_curves.png")

# 图二：横向多SNR对比图
demo_img, demo_lidar = test_dataset[-1]
orig_img_np = np.clip(demo_img.numpy().transpose(1, 2, 0), 0, 1)
orig_bev_np = demo_lidar.numpy()[1, :, :]

fig2, axes = plt.subplots(2, 1 + len(snr_test_list), figsize=(18, 6))
axes[0, 0].imshow(orig_img_np)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')
axes[1, 0].imshow(orig_bev_np, cmap='plasma')
axes[1, 0].set_title("Original LiDAR")
axes[1, 0].axis('off')

with torch.no_grad():
    for idx, snr in enumerate(snr_test_list):
        r_i, r_l = model(demo_img.unsqueeze(0).to(device), demo_lidar.unsqueeze(0).to(device), float(snr))
        img_show = np.clip(r_i.cpu().squeeze(0).numpy().transpose(1, 2, 0), 0, 1)
        lidar_show = r_l.cpu().squeeze(0).numpy()[1, :, :]
        
        p_val = psnr_calc(orig_img_np, img_show, data_range=1.0)
        
        axes[0, idx+1].imshow(img_show)
        axes[0, idx+1].set_title(f"SNR={snr}dB\nPSNR: {p_val:.1f}dB", fontsize=9)
        axes[0, idx+1].axis('off')
        
        axes[1, idx+1].imshow(lidar_show, cmap='plasma')
        axes[1, idx+1].set_title(f"SNR={snr}dB", fontsize=9)
        axes[1, idx+1].axis('off')

plt.tight_layout()
plt.savefig("visual_comparison.png") # 自动保存对比图到本地
print("[✔] 对比图已保存为 visual_comparison.png")

# 同时将两张图在屏幕上依次弹出来
plt.show()