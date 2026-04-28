import os
import glob
from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# 1️⃣ Harris 角点热力图：输出 dense cornerness response map
# ============================================================
def harris_corner_heatmap_from_pil(
    pil_img: Image.Image,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    pre_blur: bool = True,
    blur_sigma: float = 1.2,
    relu: bool = True,
    clip_percentile: float = 99.5,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    输入: PIL图像(灰度 or RGB)
    输出: [1,H,W] float32 Tensor, 范围[0,1]，Harris角点响应热力图

    说明:
    - Harris 输出是 dense response（不像 FAST 是点集），非常适合做监督/结构loss。
    - 可选 top_k：只保留响应最强的 top_k 区域（通过阈值截断），避免满图都是弱响应。
    """
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)

    if pre_blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Harris response: float32
    R = cv2.cornerHarris(img, blockSize=block_size, ksize=ksize, k=k)  # [H,W]
    # 适度膨胀，让角点响应更“团”，更利于回归
    R = cv2.dilate(R, None)

    if relu:
        R = np.maximum(R, 0.0)

    # 可选：只保留 top_k 强响应（通过取阈值）
    if top_k is not None and top_k > 0:
        # 用分位数近似控制数量（更稳定、无需复杂NMS）
        flat = R.reshape(-1)
        # 取第 top_k 大的值作为阈值
        if top_k < flat.size:
            thr = np.partition(flat, -top_k)[-top_k]
            R = np.where(R >= thr, R, 0.0)

    # 轻微平滑，让热力图更连续
    if blur_sigma is not None and blur_sigma > 0:
        R = cv2.GaussianBlur(R, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # 鲁棒归一化：按高分位截断后再归一化，避免少数极大值压扁整体对比度
    if clip_percentile is not None and 0 < clip_percentile < 100:
        hi = float(np.percentile(R, clip_percentile))
        if hi > 0:
            R = np.clip(R, 0.0, hi)
    m = float(R.max())
    if m > 0:
        R = R / (m + 1e-8)

    return torch.from_numpy(R).unsqueeze(0).float()  # [1,H,W]


# ============================================================
# 2️⃣ Dataset：直接返回 SAR_harris, OPT_harris
# ============================================================
class SAROptHarrisDataset(Dataset):
    def __init__(
        self,
        sar_dir: str,
        opt_dir: str,
        img_size: int = 256,
        block_size: int = 2,
        ksize: int = 3,
        k: float = 0.04,
        harris_top_k: Optional[int] = None,
        blur_sigma: float = 1.2,
    ):
        self.sar_files = sorted(glob.glob(os.path.join(sar_dir, "*.png")))
        self.opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.png")))
        assert len(self.sar_files) == len(self.opt_files), "数据数量不匹配"

        self.resize = transforms.Resize((img_size, img_size))
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
        self.harris_top_k = harris_top_k
        self.blur_sigma = blur_sigma

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar = Image.open(self.sar_files[idx]).convert("L")  # SAR单通道
        opt = Image.open(self.opt_files[idx]).convert("RGB")  # OPT可能是RGB；Harris内部会转灰度
        sar = self.resize(sar)
        opt = self.resize(opt)

        sar_harris = harris_corner_heatmap_from_pil(
            sar,
            block_size=self.block_size,
            ksize=self.ksize,
            k=self.k,
            blur_sigma=self.blur_sigma,
            top_k=self.harris_top_k,
        )
        opt_harris = harris_corner_heatmap_from_pil(
            opt,
            block_size=self.block_size,
            ksize=self.ksize,
            k=self.k,
            blur_sigma=self.blur_sigma,
            top_k=self.harris_top_k,
        )

        # Debug 可视化（可选）：只保存前几张，避免拖慢训练
        # save_image(sar_harris, f"/NAS_data/yjy/debug_sar_harris.png")
        # save_image(opt_harris, f"/NAS_data/yjy/debug_opt_harris.png")

        return sar_harris, opt_harris


# ============================================================
# 3️⃣ U-Net（与你原来一致）
# ============================================================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = self.block(in_channels, 64)
        self.enc2 = self.block(64, 128)
        self.enc3 = self.block(128, 256)
        self.enc4 = self.block(256, 512)
        self.bottleneck = self.block(512, 1024)
        self.up4 = self.up_block(1024, 512)
        self.up3 = self.up_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.up1 = self.up_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        d4 = self.up4(b) + e4
        d3 = self.up3(d4) + e3
        d2 = self.up2(d3) + e2
        d1 = self.up1(d2) + e1
        out = torch.sigmoid(self.final(d1))
        return out


# ============================================================
# 4️⃣ 训练入口
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Harris 参数建议（256×256 常用默认即可）
    block_size = 4   # 邻域大小
    ksize = 3        # Sobel窗口
    k = 0.04         # Harris参数
    harris_top_k = 6000  # 可选：限制只保留最强的若干响应；不想截断可设 None
    blur_sigma = 2

    train_dataset = SAROptHarrisDataset(
        sar_dir="/data/hjf/Dataset/SEN12_Scene/trainA/",
        opt_dir="/data/hjf/Dataset/SEN12_Scene/trainB/",
        img_size=256,
        block_size=block_size,
        ksize=ksize,
        k=k,
        harris_top_k=harris_top_k,
        blur_sigma=blur_sigma,
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=2, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)

    # Harris热力图是连续密集的，L1通常更稳
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 200
    save_dir = "/NAS_data/yjy/DDBM_GT_Unet/DDBM_S2O_HARRIS_corner_results/SEN_scene"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for sar_harris, opt_harris in progress_bar:
            sar_harris = sar_harris.to(device, non_blocking=True)
            opt_harris = opt_harris.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(sar_harris)
            loss = criterion(pred, opt_harris)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sar_harris, opt_harris = next(iter(train_loader))
                sar_harris = sar_harris.to(device)
                opt_harris = opt_harris.to(device)
                pred = model(sar_harris)

            sar_harris, opt_harris, pred = sar_harris.cpu(), opt_harris.cpu(), pred.cpu()
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            for i in range(3):
                axes[i, 0].imshow(sar_harris[i][0], cmap='gray')
                axes[i, 0].set_title('Input SAR-Harris')
                axes[i, 1].imshow(pred[i][0].detach(), cmap='gray')
                axes[i, 1].set_title('Predicted Harris')
                axes[i, 2].imshow(opt_harris[i][0], cmap='gray')
                axes[i, 2].set_title('GT Optical-Harris')
                for j in range(3):
                    axes[i, j].axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/epoch_{epoch + 1}.png")
            plt.close()

            torch.save(model.state_dict(), f"{save_dir}/unet_epoch_{epoch + 1}.pth")

    final_path = os.path.join(save_dir, "unet_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training finished! Final model saved to: {final_path}")
