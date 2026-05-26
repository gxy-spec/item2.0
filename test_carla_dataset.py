import io
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image

print("========= 正在从 Hugging Face 自动下载/加载轻量化多模态 CARLA 数据集 =========")
# 1. 自动流式加载数据集（stream=True可以让你只抓取最前面几条，不用把整个多G的合集全下完，极度轻量省时！）

dataset = load_dataset("immanuelpeter/carla-autopilot-multimodal-dataset", split="train", streaming=True)
sample = next(iter(dataset))
print("[✔] 成功获取多模态样本数据！")

# ========================================================
# 🎥 【模态一】：直接提取正前方相机 RGB 图像
# ========================================================
# 核心修正：datasets 已经自动帮我们解压成 PIL 图片对象了，直接赋值即可！
image_rgb = sample['image_front']
print(f" -> 模态一（图像）：成功提取前向相机图像，分辨率 {image_rgb.size}，类型 {type(image_rgb)}")

# ========================================================
# 🛰️ 【模态二】：解析并恢复 3D LiDAR 激光点云
# ========================================================
# 因为 'lidar' 在该数据集中已经是处理好的嵌套列表，直接转换为 numpy 矩阵
lidar_data = np.array(sample['lidar'], dtype=np.float32) # 形状为 (N, 4) 或 (N, 3)
print(f" -> 模态二（点云）：成功读取点云，包含 {lidar_data.shape[0]} 个 3D 探测点")

# ========================================================
# 📝 【模态三标签】：打印用于论文映射的环境与控制特征
# ========================================================
print("\n--- 采集到的环境、状态与控制物理量（完美支撑你的模态三明文短语） ---")
print(f"当前车速: {sample['speed_kmh']:.2f} km/h")
print(f"方向盘角度 (Steer): {sample['steer']:.4f}")
print(f"油门开度 (Throttle): {sample['throttle']:.4f}")
print(f"刹车开度 (Brake): {sample['brake']:.4f}")
print(f"当前天气状态 -> 雾气浓度: {sample['weather_fog_density']:.1f}%, 降水量: {sample['weather_precipitation']:.1f}%")

# ========================================================
# 📊 【双模态一键多功能可视化】
# ========================================================
fig = plt.figure(figsize=(15, 6))

# 子图1：展示主车前向视角的彩色图像
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(image_rgb)
ax1.set_title("Modality 1: CARLA Front RGB Camera View", fontsize=12, fontweight="bold")
ax1.axis('off')

# 子图2：展示 3D 激光雷达点云（BEV 鸟瞰投影俯视图）
ax2 = fig.add_subplot(1, 2, 2)
# 提取雷达点的 X (前后距离) 和 Y (左右距离)
x_points = lidar_data[:, 0]
y_points = lidar_data[:, 1]
z_points = lidar_data[:, 2] # 将高度作为颜色映射

# 绘制点云散点图（设定合理的 bird's-eye view 显示范围，视觉效果最好）
scatter = ax2.scatter(y_points, x_points, c=z_points, cmap='jet', s=1, alpha=0.5)
ax2.set_xlim(-20, 20)
ax2.set_ylim(0, 40)
ax2.set_xlabel("Left / Right Distance (m)")
ax2.set_ylabel("Forward Distance (m)")
ax2.set_title("Modality 2: 3D LiDAR Point Cloud (Bird's Eye View)", fontsize=12, fontweight="bold")
fig.colorbar(scatter, ax=ax2, label="Height Z (m)")

plt.suptitle("CARLA Autopilot Multimodal Sensor Stream Preview", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.show()