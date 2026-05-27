import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class KittiLidarBEVDataset(Dataset):
    def __init__(self, base_path="data\\2011_09_26_drive_0001_sync", grid_size=(256, 256)):
        """
        KITTI 点云鸟瞰图单模态数据集
        :param grid_size: 鸟瞰图的分辨率 (H, W)，推荐 256x256 
        """
        self.lidar_dir = os.path.join(base_path, "velodyne_points", "data")
        self.lidar_files = sorted(glob.glob(os.path.join(self.lidar_dir, "*.bin")))
        self.grid_size = grid_size
        
        if len(self.lidar_files) == 0:
            raise FileNotFoundError(f"❌ 未在 {self.lidar_dir} 中找到 .bin 文件！")

    def __len__(self):
        return len(self.lidar_files)

    def __getitem__(self, idx):
        # 1. 读取原始 3D 点云 [X, Y, Z, Intensity]
        scan = np.fromfile(self.lidar_files[idx], dtype=np.float32)
        points = scan.reshape((-1, 4))
        
        # 2. 设定鸟瞰图的物理感知范围 (以车头为中心，过滤掉无用远景)
        x_range = (0, 40)    # 前方 0 到 40 米
        y_range = (-20, 20)  # 左右各 20 米
        
        x_pts = points[:, 0]
        y_pts = points[:, 1]
        z_pts = points[:, 2]
        i_pts = points[:, 3]
        
        # 过滤在范围内的点
        mask = (x_pts >= x_range[0]) & (x_pts <= x_range[1]) & \
               (y_pts >= y_range[0]) & (y_pts <= y_range[1])
        x_pts, y_pts, z_pts, i_pts = x_pts[mask], y_pts[mask], z_pts[mask], i_pts[mask]
        
        # 3. 将物理坐标映射到像素网格坐标 [256, 256]
        x_grid = (((x_pts - x_range[0]) / (x_range[1] - x_range[0])) * (self.grid_size[0] - 1)).astype(np.int32)
        y_grid = (((y_pts - y_range[0]) / (y_range[1] - y_range[0])) * (self.grid_size[1] - 1)).astype(np.int32)
        
        # 翻转 X 轴让前方朝上
        x_grid = (self.grid_size[0] - 1) - x_grid
        
        # 4. 创建双通道 BEV 图像: 通道0存放高度(Z)，通道1存放反射强度(I)
        bev_map = np.zeros((2, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        
        # 填充数据（由于同一格可能有多个点，后读入的会覆盖，作为基础 Baseline 足够）
        bev_map[0, x_grid, y_grid] = z_pts  # 高度特征
        bev_map[1, x_grid, y_grid] = i_pts  # 强度特征
        
        return torch.tensor(bev_map)

if __name__ == "__main__":
    dataset = KittiLidarBEVDataset()
    print(f"[✔] Lidar Dataset 初始化成功，总帧数: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    first_batch = next(iter(loader))
    print(f" -> 点云 BEV Batch 形状: {first_batch.shape}") # 预期: [4, 2, 256, 256]