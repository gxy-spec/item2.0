import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_calc

# 导入你现有的多模态数据集类 (自动匹配单模态的读取)
from kitti_multimodal_dataset import KittiMultiModalDataset

# ========================================================
# 1. 📡 严格与模态 1 对齐的 MMSE 无线复合信道层
# ========================================================
class A2GComplexChannelMode2(nn.Module):
    def __init__(self, k_factor=3.0):
        super(A2GComplexChannelMode2, self).__init__()
        self.k_factor = k_factor 

    def forward(self, x, snr_db):
        rms = torch.sqrt(torch.mean(x ** 2) + 1e-4)
        x_normalized = x / rms
        
        los_weight = np.sqrt(self.k_factor / (self.k_factor + 1))
        nlos_weight = np.sqrt(1 / (self.k_factor + 1))
        H = torch.ones_like(x_normalized) * los_weight + torch.randn_like(x_normalized) * nlos_weight
        
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = np.sqrt(1.0 / snr_linear)
        noise = torch.randn_like(x_normalized) * noise_std
        
        y_received = H * x_normalized + noise
        
        # 使用 MMSE 均衡算法，彻底斩断 nan
        y_equalized = (y_received * H) / (H ** 2 + (1.0 / snr_linear) + 1e-4)
        return y_equalized

# ========================================================
# 2. 🧬 模态 2：两个独立的单模态 JSCC 网络 (对齐你上传的架构)
# ========================================================
class ImageJSCCMode2(nn.Module):
    def __init__(self):
        super(ImageJSCCMode2, self).__init__()
        self.channel = A2GComplexChannelMode2()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x, snr_db):
        feat = self.encoder(x)
        feat_ch = self.channel(feat, snr_db)
        return self.decoder(feat_ch)

class LidarJSCCMode2(nn.Module):
    def __init__(self):
        super(LidarJSCCMode2, self).__init__()
        self.channel = A2GComplexChannelMode2()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
    def forward(self, x, snr_db):
        feat = self.encoder(x)
        feat_ch = self.channel(feat, snr_db)
        return self.decoder(feat_ch)

# ========================================================
# 🚀 训练驱动控制 (双基线同步并行训练)
# ========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_dataset = KittiMultiModalDataset()
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 实例化两个独立的单模态网络
img_model = ImageJSCCMode2().to(device)
lidar_model = LidarJSCCMode2().to(device)

criterion = nn.MSELoss()
optimizer_img = optim.Adam(img_model.parameters(), lr=0.0003)
optimizer_lidar = optim.Adam(lidar_model.parameters(), lr=0.0003)

epochs = 30
print("正在开启模态 2：图像与雷达双单模态基线同步训练...")

for epoch in range(epochs):
    img_model.train()
    lidar_model.train()
    total_img_loss, total_lidar_loss = 0.0, 0.0
    
    for batch_img, batch_lidar in train_loader:
        batch_img, batch_lidar = batch_img.to(device), batch_lidar.to(device)
        
        # 动态信道热身
        random_snr = np.random.uniform(12.0, 20.0) if epoch < 5 else np.random.uniform(0.0, 20.0)
        
        # 优化图像单模态
        optimizer_img.zero_grad()
        recon_img = img_model(batch_img, random_snr)
        loss_img = criterion(recon_img, batch_img)
        loss_img.backward()
        optimizer_img.step()
        total_img_loss += loss_img.item() * batch_img.size(0)
        
        # 优化雷达单模态
        optimizer_lidar.zero_grad()
        recon_lidar = lidar_model(batch_lidar, random_snr)
        loss_lidar = criterion(recon_lidar, batch_lidar)
        loss_lidar.backward()
        optimizer_lidar.step()
        total_lidar_loss += loss_lidar.item() * batch_lidar.size(0)
        
    print(f"Epoch [{epoch+1}/{epochs}] | Img Loss: {total_img_loss/len(train_dataset):.5f} | Lidar Loss: {total_lidar_loss/len(train_dataset):.5f}")

# ─── ⏱️ 需求 1：保存模型时加上时间戳 ───
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(img_model.state_dict(), f"model_mode2_image_{timestamp}.pth")
torch.save(lidar_model.state_dict(), f"model_mode2_lidar_{timestamp}.pth")
print(f"[✔] 两个单模态模型已成功保存，附带时间戳：_{timestamp}")

# ========================================================
# 📊 测试扫频：计算两个单模态随信噪比变化的学术曲线
# ========================================================
img_model.eval()
lidar_model.eval()
snr_test_list = [-5, 0, 5, 10, 15, 20]
img_psnr_results = []
lidar_nmse_results = []

print("\n正在对模态 2 两个独立基线进行信道扫频测试...")
with torch.no_grad():
    for test_snr in snr_test_list:
        current_psnr, current_nmse = [], []
        for t_img, t_lidar in test_loader:
            t_img, t_lidar = t_img.to(device), t_lidar.to(device)
            
            # 独立测试
            r_img = img_model(t_img, test_snr)
            r_lidar = lidar_model(t_lidar, test_snr)
            
            img_org = np.clip(t_img.cpu().squeeze(0).numpy().transpose(1, 2, 0), 0, 1)
            img_rec = np.clip(r_img.cpu().squeeze(0).numpy().transpose(1, 2, 0), 0, 1)
            lidar_org = t_lidar.cpu().squeeze(0).numpy()
            lidar_rec = r_lidar.cpu().squeeze(0).numpy()
            
            current_psnr.append(psnr_calc(img_org, img_rec, data_range=1.0))
            current_nmse.append(np.mean((lidar_org - lidar_rec) ** 2) / (np.mean(lidar_org ** 2) + 1e-8))
            
        img_psnr_results.append(np.mean(current_psnr))
        lidar_nmse_results.append(np.mean(current_nmse))

# ========================================================
# 🎨 终极绘图系统：单画布统一编排 + 自动打上时间戳
# ========================================================
num_cols = 1 + len(snr_test_list)
fig = plt.figure(figsize=(18, 10))

# ─── 1. 学术折线图 (第一行) ───
ax1 = plt.subplot2grid((3, num_cols), (0, 0), colspan=2)
ax1.plot(snr_test_list, img_psnr_results, 'ro-', linewidth=2)
ax1.set_xlabel('Channel SNR (dB)')
ax1.set_ylabel('Image PSNR (dB)')
ax1.set_title('Mode 2: Image-Only Baseline PSNR')
ax1.grid(True)

ax2 = plt.subplot2grid((3, num_cols), (0, 2), colspan=2)
ax2.plot(snr_test_list, lidar_nmse_results, 'g^-', linewidth=2)
ax2.set_xlabel('Channel SNR (dB)')
ax2.set_ylabel('LiDAR BEV NMSE')
ax2.set_title('Mode 2: LiDAR-Only Baseline NMSE')
ax2.grid(True)

# ─── 2. 多 SNR 独立重构对比 (第二、三行) ───
demo_img, demo_lidar = test_dataset[-1]
orig_img_np = np.clip(demo_img.numpy().transpose(1, 2, 0), 0.0, 1.0)
orig_bev_np = demo_lidar.numpy()[1, :, :]

# 第 0 列真值
ax_i_org = plt.subplot2grid((3, num_cols), (1, 0))
ax_i_org.imshow(orig_img_np)
ax_i_org.set_title("Original Image", fontsize=9, fontweight="bold")
ax_i_org.axis('off')

ax_l_org = plt.subplot2grid((3, num_cols), (2, 0))
ax_l_org.imshow(orig_bev_np, cmap='plasma')
ax_l_org.set_title("Original LiDAR", fontsize=9, fontweight="bold")
ax_l_org.axis('off')

# 渲染各 SNR 独立解码后的画面
with torch.no_grad():
    for idx, snr in enumerate(snr_test_list):
        r_i = img_model(demo_img.unsqueeze(0).to(device), float(snr))
        r_l = lidar_model(demo_lidar.unsqueeze(0).to(device), float(snr))
        
        img_show = np.clip(r_i.cpu().squeeze(0).numpy().transpose(1, 2, 0), 0.0, 1.0)
        lidar_show = r_l.cpu().squeeze(0).numpy()[1, :, :]
        p_val = psnr_calc(orig_img_np, img_show, data_range=1.0)
        
        col_pos = idx + 1
        # 图像独立重构 (第二行)
        ax_i_rec = plt.subplot2grid((3, num_cols), (1, col_pos))
        ax_i_rec.imshow(img_show)
        ax_i_rec.set_title(f"SNR={snr}dB\nPSNR:{p_val:.1f}dB", fontsize=9)
        ax_i_rec.axis('off')
        
        # 雷达独立重构 (第三行)
        ax_l_rec = plt.subplot2grid((3, num_cols), (2, col_pos))
        ax_l_rec.imshow(lidar_show, cmap='plasma')
        ax_l_rec.set_title(f"SNR={snr}dB", fontsize=9)
        ax_l_rec.axis('off')

plt.suptitle(f"KITTI Mode 2 (Single-Modal Baselines Evaluation) [{timestamp}]", fontsize=12, fontweight="bold")
plt.tight_layout()

# ─── ⏱️ 需求 2：保存图表时加上时间戳 ───
image_save_name = f"mode2_baselines_report_{timestamp}.png"
plt.savefig(image_save_name, dpi=300)
print(f"[✔] 模态 2 学术对比图已安全导出：{image_save_name}")

plt.show()