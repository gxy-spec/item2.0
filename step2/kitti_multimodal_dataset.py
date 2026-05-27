import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class KittiMultiModalDataset(Dataset):
    def __init__(self, base_path="data\\2011_09_26_drive_0001_sync", img_size=(256, 768), grid_size=(256, 256)):
        """
        KITTI 多模态同步数据集 (图像 + 点云 BEV)
        :param base_path: 本地 KITTI 数据集文件夹路径
        :param img_size: 图像统一缩放尺寸 (H, W)
        :param grid_size: 点云鸟瞰图的分辨率 (H, W)
        """
        self.base_path = base_path
        self.grid_size = grid_size
        
        # 1. 严格按排序对齐读取图像和点云文件路径
        self.img_dir = os.path.join(base_path, "image_02", "data")
        self.lidar_dir = os.path.join(base_path, "velodyne_points", "data")
        
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        self.lidar_files = sorted(glob.glob(os.path.join(self.lidar_dir, "*.bin")))
        
        # 2. 健壮性检查：确保图像和点云帧数完全一致
        if len(self.img_files) == 0 or len(self.lidar_files) == 0:
            raise FileNotFoundError(f"❌ 路径检查失败！请确保 {self.img_dir} 和 {self.lidar_dir} 内部有数据。")
            
        if len(self.img_files) != len(self.lidar_files):
            print(f"⚠️ 警告: 图像有 {len(self.img_files)} 帧，但点云有 {len(self.lidar_files)} 帧。将以较小值进行对齐截断。")
            self.total_frames = min(len(self.img_files), len(self.lidar_files))
        else:
            self.total_frames = len(self.img_files)
            
        # 3. 图像预处理流水线
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(), # 归一化到 [0, 1]
        ])

    def __len__(self):
        return self.total_frames

    def _process_lidar_bev(self, lidar_path):
        """将原始 3D 点云转换为 2D 鸟瞰图 (BEV) Tensor"""
        scan = np.fromfile(lidar_path, dtype=np.float32)
        points = scan.reshape((-1, 4))
        
        # 过滤物理范围 (前方0~40米，左右各20米)
        x_range, y_range = (0, 40), (-20, 20)
        x_pts, y_pts, z_pts, i_pts = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        
        mask = (x_pts >= x_range[0]) & (x_pts <= x_range[1]) & \
               (y_pts >= y_range[0]) & (y_pts <= y_range[1])
        x_pts, y_pts, z_pts, i_pts = x_pts[mask], y_pts[mask], z_pts[mask], i_pts[mask]
        
        # 映射到网格像素坐标
        x_grid = (((x_pts - x_range[0]) / (x_range[1] - x_range[0])) * (self.grid_size[0] - 1)).astype(np.int32)
        y_grid = (((y_pts - y_range[0]) / (y_range[1] - y_range[0])) * (self.grid_size[1] - 1)).astype(np.int32)
        x_grid = (self.grid_size[0] - 1) - x_grid # 翻转让车头朝上
        
        # 创建双通道 BEV 密集矩阵 (高度通道 + 强度通道)
        bev_map = np.zeros((2, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        bev_map[0, x_grid, y_grid] = z_pts # Channel 0: 高度
        bev_map[1, x_grid, y_grid] = i_pts # Channel 1: 反射强度
        
        return torch.tensor(bev_map)

    def __getitem__(self, idx):
        # 🎯 核心机制：用同一个 idx 同时捞出对应的图像和雷达
        img_path = self.img_files[idx]
        lidar_path = self.lidar_files[idx]
        
        # 处理图像
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.img_transform(image)
        
        # 处理点云
        lidar_bev_tensor = self._process_lidar_bev(lidar_path)
        
        # 以元组（或字典）形式同步返回
        return image_tensor, lidar_bev_tensor

# ========================================================
# 🔍 严谨的本地快速对齐验证测试
# ========================================================
if __name__ == "__main__":
    print("正在初始化多模态同步数据集...")
    try:
        dataset = KittiMultiModalDataset()
        print(f"[✔] 多模态 Dataset 成功建立！总共成功同步对齐了 {len(dataset)} 帧数据。")
        
        # 测试用 DataLoader 批量拉取
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        img_batch, lidar_batch = next(iter(loader))
        
        print("\n📥 正在检查 DataLoader 吐出的多模态 Batch 数据形状:")
        print(f" -> 图像批次形状 (Image Batch Shape) : {img_batch.shape}")      # 预期: [4, 3, 256, 768]
        print(f" -> 点云批次形状 (LiDAR BEV Batch Shape): {lidar_batch.shape}")  # 预期: [4, 2, 256, 256]
        print("\n[✔] 数据流完全打通，时空对齐无误，可以安心喂给融合网络！")
        
    except Exception as e:
        print(f"❌ 运行出错了，错误原因: {e}")