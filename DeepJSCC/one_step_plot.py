import json
import os
import matplotlib.pyplot as plt

# JSON 文件路径
json_path = "./results/evaluation_deep_jscc_cifar10_snr10_awgn_kn0.0833.json"

# 从文件名中提取 compression_ratio
compression_ratio = None
base_name = os.path.basename(json_path)
try:
    compression_ratio = float(base_name.split('_kn')[1].split('.json')[0])
except (IndexError, ValueError):
    pass

# 读取 JSON
with open(json_path, "r") as f:
    data = json.load(f)

# 提取 AWGN 数据
awgn_data = data["awgn"]

# 提取 PSNR 和 SSIM
psnr_dict = awgn_data["psnr"]
ssim_dict = awgn_data["ssim"]

# 转换成列表
snr_values = [int(k) for k in psnr_dict.keys()]
psnr_values = [float(v) for v in psnr_dict.values()]
ssim_values = [float(v) for v in ssim_dict.values()]

# =========================
# 图1：SNR vs PSNR
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    snr_values,
    psnr_values,
    marker='o',
    linewidth=2
)

plt.xlabel("SNR (dB)")
plt.ylabel("PSNR (dB)")
psnr_title = "SNR vs PSNR"
if compression_ratio is not None:
    psnr_title += f" (compression_ratio={compression_ratio:.4f})"
plt.title(psnr_title)
plt.grid(True)

# 保存图片
plt.savefig("./results/snr_vs_psnr.png")

plt.show()

# =========================
# 图2：SNR vs SSIM
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    snr_values,
    ssim_values,
    marker='s',
    linewidth=2
)

plt.xlabel("SNR (dB)")
plt.ylabel("SSIM")
ssim_title = "SNR vs SSIM"
if compression_ratio is not None:
    ssim_title += f" (compression_ratio={compression_ratio:.4f})"
plt.title(ssim_title)
plt.grid(True)

# 保存图片
plt.savefig("./results/snr_vs_ssim.png")

plt.show()

print("Plots saved successfully!")