import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    hsi_path = os.path.join(current_dir, "Italy_hsi.mat")
    lidar_path = os.path.join(current_dir, "Italy_lidar.mat")
    gt_path = os.path.join(current_dir, "allgrd.mat")

    print("正在分别读取 Italy HSI / LiDAR / GT 三个文件...")

    data_hsi = sio.loadmat(hsi_path)
    data_lidar = sio.loadmat(lidar_path)
    data_gt = sio.loadmat(gt_path)

    hsi = data_hsi["data"]          # (166, 600, 63)
    lidar = data_lidar["data"]      # (166, 600, 2)
    gt = data_gt["mask_test"]       # (166, 600)

    print(f"HSI shape: {hsi.shape}")
    print(f"LiDAR shape: {lidar.shape}")
    print(f"GT shape: {gt.shape}")

    fig = plt.figure(figsize=(16, 10))

    # 1. HSI pseudo-color visualization
    band_r = min(30, hsi.shape[2] - 1)
    band_g = min(15, hsi.shape[2] - 1)
    band_b = min(5, hsi.shape[2] - 1)
    rgb_img = np.stack([hsi[:, :, band_r], hsi[:, :, band_g], hsi[:, :, band_b]], axis=-1)
    rgb_min = rgb_img.min()
    rgb_max = rgb_img.max()
    rgb_img = (rgb_img - rgb_min) / (rgb_max - rgb_min + 1e-8)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(rgb_img)
    ax1.set_title(f"Italy HSI Pseudo Color (Bands: {band_r}, {band_g}, {band_b})", fontsize=13)
    ax1.axis("off")

    # 2. First LiDAR channel
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(lidar[:, :, 0], cmap="gray")
    ax2.set_title("Italy LiDAR Channel 0", fontsize=13)
    ax2.axis("off")

    # 3. Second LiDAR channel, if available
    ax3 = fig.add_subplot(2, 2, 3)
    lidar_channel_idx = 1 if lidar.shape[2] > 1 else 0
    ax3.imshow(lidar[:, :, lidar_channel_idx], cmap="terrain")
    ax3.set_title(f"Italy LiDAR Channel {lidar_channel_idx}", fontsize=13)
    ax3.axis("off")

    # 4. 3D surface from one LiDAR channel
    crop_h, crop_w = min(120, lidar.shape[0]), min(120, lidar.shape[1])
    lidar_crop = lidar[:crop_h, :crop_w, 0]
    x = np.arange(crop_w)
    y = np.arange(crop_h)
    x, y = np.meshgrid(x, y)

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    surf = ax4.plot_surface(x, y, lidar_crop, cmap="terrain", edgecolor="none", rstride=2, cstride=2)
    ax4.set_title("Italy LiDAR 3D Surface", fontsize=13)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Height")
    ax4.view_init(elev=35, azim=45)
    fig.colorbar(surf, ax=ax4, shrink=0.5, aspect=10, label="LiDAR Value")

    plt.tight_layout()
    output_name = "italy_multimodal_visualization.png"
    plt.savefig(output_name, dpi=300)

    print(f"\n[OK] 运行成功！Italy 多模态可视化结果已保存到: '{output_name}'")
    plt.show()


if __name__ == "__main__":
    main()
