import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ========================================================
# 1. ⚙️ 路径精确定位 (完美对应你的目录结构)
# ========================================================
base_path = "data\\2011_09_26_drive_0001_sync"
img_dir = os.path.join(base_path, "image_02", "data")       # 左侧彩色相机
lidar_dir = os.path.join(base_path, "velodyne_points", "data") # 64线激光雷达
oxts_dir = os.path.join(base_path, "oxts", "data")           # 惯导/GPS数据

# 严格排序，确保时间戳一帧一帧完全对齐
img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.bin")))
oxts_files = sorted(glob.glob(os.path.join(oxts_dir, "*.txt")))

total_frames = min(len(img_files), len(lidar_files), len(oxts_files))
print(f"[✔] KITTI 序列对齐成功！检测到可用同步总帧数: {total_frames}")

# ========================================================
# 2. 🔄 核心读取逻辑：提取特定一帧的三个模式所需数据
# ========================================================
def fetch_kitti_multimodal_frame(frame_idx):
    if frame_idx >= total_frames:
        raise IndexError(f"帧索引超限，当前最大可用帧为 {total_frames-1}")
        
    # --- 【模式一/二前端：2D 图像读取】 ---
    img = Image.open(img_files[frame_idx])
    
    # --- 【模式一/二前端：3D 点云读取】 ---
    # np.fromfile 读取成一维数组，再重塑成 (N, 4) 矩阵 [X, Y, Z, Intensity]
    pts = np.fromfile(lidar_files[frame_idx], dtype=np.float32).reshape((-1, 4))
    
    # --- 【模式三前端：读取惯导数据并直译为抽象语义文本】 ---
    with open(oxts_files[frame_idx], 'r') as f:
        oxts_line = f.readline().split()
    
    # 提取关键物理量（KITTI 官方规范定义）
    lat = float(oxts_line[0])   # 纬度
    lon = float(oxts_line[1])   # 经度
    vf = float(oxts_line[8]) * 3.6  # 第9项：正向速度，转换为 km/h
    af = float(oxts_line[11])       # 第12项：正向加速度 m/s^2
    
    # 在发射端进行超高语义抽象文本提炼（模拟你的模式三）
    motion_status = "Cruising"
    if af < -1.5: motion_status = "Decelerating/Braking"
    elif af > 1.5: motion_status = "Accelerating"
    
    # 根据点云简单统计前方 5m-15m 范围内是否有紧密反射点
    front_cluster = pts[(pts[:, 0] > 5) & (pts[:, 0] < 15) & (np.abs(pts[:, 1]) < 2.5)]
    obstacle_desc = "Obstacle detected ahead" if len(front_cluster) > 80 else "Forward pathway clear"
    
    # 生成模式三极端恶劣信道下的生存明文
    mode3_text = f"GPS:({lat:.4f},{lon:.4f}) -> {motion_status} ({vf:.1f} km/h) -> {obstacle_desc}."
    
    return img, pts, mode3_text

# ========================================================
# 3. 📊 测试读取第 0 帧并进行全模态可视化展示
# ========================================================
current_frame = 0
image_data, lidar_data, text_data = fetch_kitti_multimodal_frame(current_frame)

print(f"\n成功解析第 {current_frame} 帧数据：")
print(f" ├─ 图像分辨率: {image_data.size}")
print(f" ├─ 64线雷达探测点数: {lidar_data.shape[0]} 个")
print(f" └─ 模式三发射端直译文本: \"{text_data}\"")

# 弹窗绘图
fig = plt.figure(figsize=(15, 6))

# 左半边：展示模式一/二所需的 2D 彩色感知画面
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(image_data)
ax1.set_title("Modality 1: Camera Image (image_02)", fontsize=11, fontweight="bold")
ax1.axis('off')

# 右半边：展示模式一/二所需的 3D 激光点云（鸟瞰俯视图 BEV）
ax2 = fig.add_subplot(1, 2, 2)
x_pts = lidar_data[:, 0]  # 前后
y_pts = lidar_data[:, 1]  # 左右
intensity = lidar_data[:, 3]  # 反射强度

scatter = ax2.scatter(y_pts, x_pts, c=intensity, cmap='plasma', s=0.1, alpha=0.6)
ax2.set_xlim(-15, 15)
ax2.set_ylim(0, 45)
ax2.set_xlabel("Left / Right (Y in meters)")
ax2.set_ylabel("Forward (X in meters)")
ax2.set_title("Modality 2: Velodyne 64-Line LiDAR BEV Plot", fontsize=11, fontweight="bold")
fig.colorbar(scatter, ax=ax2, label="Reflective Intensity")

# 顶部横幅：霸气展示你的模式三“超高语义抽象文本”
plt.suptitle(f"KITTI Native Sequence Multi-Modal Framework\n[Mode 3 High-Abstract Text]: {text_data}", 
             fontsize=12, color='darkgreen', fontweight="bold", y=0.98)
plt.tight_layout()
plt.show()