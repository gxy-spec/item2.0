import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class KittiImageDataset(Dataset):
    def __init__(self, base_path="data\\2011_09_26_drive_0001_sync", img_size=(256, 768)):
        """
        KITTI 图像单模态数据集
        :param img_size: 缩放后的统一分辨率 (H, W)，推荐使用 256x768 保持宽幅比例
        """
        self.img_dir = os.path.join(base_path, "image_02", "data")
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        
        if len(self.img_files) == 0:
            raise FileNotFoundError(f"❌ 未在 {self.img_dir} 中找到 .png 图片，请检查路径！")
            
        # 图像预处理流水线
        self.transform = transforms.Compose([
            transforms.Resize(img_size),          # 缩放到统一大小
            transforms.ToTensor(),                # 转换为 Tensor 且像素值归一化到 [0, 1]
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 暂不标准化，方便直接还原可视化
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB") # 确保是 3 通道彩色图
        image_tensor = self.transform(image)
        return image_tensor

# 本地快速验证测试
if __name__ == "__main__":
    dataset = KittiImageDataset()
    print(f"[✔] Dataset 初始化成功，总帧数: {len(dataset)}")
    
    # 用 DataLoader 试装载一个 Batch
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    first_batch = next(iter(loader))
    print(f" -> 一个 Batch 的图像 Tensor 形状: {first_batch.shape}") # 预期: [4, 3, 256, 768]