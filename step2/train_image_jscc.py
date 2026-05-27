import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from kitti_dataset import KittiImageDataset

# ========================================================
# 1. 🧬 搭建极简的联合源信道图像自编码网络 (JSCC Architecture)
# ========================================================
class ImageJSCC(nn.Module):
    def __init__(self):
        super(ImageJSCC, self).__init__()
        
        # 编码器：把 [B, 3, 256, 768] 压缩
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # -> [B, 16, 128, 384]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # -> [B, 32, 64, 192]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),  # -> [B, 32, 32, 96] (极度压缩的语义特征空间)
            nn.ReLU()
        )
        
        # 解码器：从低维特征还原回原图
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1), # -> [B, 32, 64, 192]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),# -> [B, 16, 128, 384]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1), # -> [B, 3, 256, 768]
            nn.Sigmoid() # 保证输出的像素值严格在 [0, 1] 之间
        )

    def forward(self, x):
        # 现阶段：不加无线信道噪声，走纯粹的闭环重构验证
        latent_features = self.encoder(x)
        reconstructed_x = self.decoder(latent_features)
        return reconstructed_x

# ========================================================
# 2. 🚀 训练主流程
# ========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用的计算设备: {device}")

# 加载数据
dataset = KittiImageDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = ImageJSCC().to(device)
criterion = nn.MSELoss() # 重构误差用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
print("开始执行图像模态单独训练测试...")

for epoch in range(epochs):
    total_loss = 0.0
    for batch_img in loader:
        batch_img = batch_img.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_img)
        loss = criterion(outputs, batch_img)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_img.size(0)
        
    print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {total_loss / len(dataset):.6f}")

print("[✔] 训练初步完成！正在现场对比重构效果...")

# ========================================================
# 3. 📊 直观验证：抽取最后一帧看看重构质量
# ========================================================
model.eval()
with torch.no_grad():
    test_img = dataset[0].unsqueeze(0).to(device) # 取第一帧做测试
    recon_img = model(test_img).cpu().squeeze(0).numpy()
    orig_img = test_img.cpu().squeeze(0).numpy()

# 调整通道顺序从 [C, H, W] 到 [H, W, C] 以供 matplotlib 画图
orig_img = orig_img.transpose(1, 2, 0)
recon_img = recon_img.transpose(1, 2, 0)

# 画图对比
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].imshow(orig_img)
axes[0].set_title("Original Input Image (From KITTI)", fontsize=10, fontweight="bold")
axes[0].axis('off')

axes[1].imshow(recon_img)
axes[1].set_title("Reconstructed Semantic Image (No Noise Baseline)", fontsize=10, fontweight="bold")
axes[1].axis('off')

plt.tight_layout()
plt.show()