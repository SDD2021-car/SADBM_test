import os
import glob
from typing import List, Optional

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
# 1️⃣ LoG 特征提取：Gaussian + Laplacian (Laplacian of Gaussian)
#    输出 dense blob/salient response map（适合训练）
# ============================================================
def log_response_map_from_pil(
    pil_img: Image.Image,
    sigmas: List[float] = [1.0, 2.0],   # 建议两尺度即可
    lap_ksize: int = 3,                 # Laplacian 核大小(1/3/5/7...)，3 常用
    pre_blur: bool = False,             # 额外预平滑（一般不必，LoG 本身包含 Gaussian）
    pre_blur_ksize: int = 3,
    take_abs: bool = True,              # LoG 正负响应 → 取绝对值更稳
    clip_percentile: float = 99.5,      # 截断极端值，避免少数峰值压扁整体
    normalize: bool = True,             # 归一化到[0,1]
) -> torch.Tensor:
    """
    输入: PIL图像(灰度 or RGB)
    输出: [1,H,W] float32 Tensor, 范围[0,1] 的 LoG 响应热力图

    说明：
    - LoG 在 blob/强散射点/局部显著细节中心处有强响应
    - 多尺度 sigmas 用 max 融合，增强尺度鲁棒性
    """
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img.astype(np.float32)

    if pre_blur:
        img = cv2.GaussianBlur(img, (pre_blur_ksize, pre_blur_ksize), 0)

    responses = []
    for sigma in sigmas:
        sm = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        lap = cv2.Laplacian(sm, ddepth=cv2.CV_32F, ksize=lap_ksize)
        responses.append(lap)

    R = np.max(np.stack(responses, axis=0), axis=0)

    if take_abs:
        R = np.abs(R)

    if clip_percentile is not None and 0 < clip_percentile < 100:
        hi = float(np.percentile(R, clip_percentile))
        if hi > 0:
            R = np.clip(R, 0.0, hi)

    if normalize:
        m = float(R.max())
        if m > 0:
            R = R / (m + 1e-8)

    return torch.from_numpy(R).unsqueeze(0).float()


# ============================================================
# 2️⃣ Dataset：返回 SAR_LoG, OPT_LoG
# ============================================================
class SAROptLoGDataset(Dataset):
    def __init__(
        self,
        sar_dir: str,
        opt_dir: str,
        img_size: int = 256,
        sigmas: List[float] = [1.0, 2.0],
        lap_ksize: int = 3,
        clip_percentile: float = 99.5,
    ):
        self.sar_files = sorted(glob.glob(os.path.join(sar_dir, "*.png")))
        self.opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.png")))
        assert len(self.sar_files) == len(self.opt_files), "数据数量不匹配"

        self.resize = transforms.Resize((img_size, img_size))
        self.sigmas = sigmas
        self.lap_ksize = lap_ksize
        self.clip_percentile = clip_percentile

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar = Image.open(self.sar_files[idx]).convert("L")
        opt = Image.open(self.opt_files[idx]).convert("RGB")

        sar = self.resize(sar)
        opt = self.resize(opt)
        sar_raw = transforms.ToTensor()(sar)  # [1,H,W] in [0,1]
        opt_raw = transforms.ToTensor()(opt)  # [3,H,W] in [0,1]
        sar_log = log_response_map_from_pil(
            sar,
            sigmas=self.sigmas,
            lap_ksize=self.lap_ksize,
            clip_percentile=self.clip_percentile,
        )
        opt_log = log_response_map_from_pil(
            opt,
            sigmas=self.sigmas,
            lap_ksize=self.lap_ksize,
            clip_percentile=self.clip_percentile,
        )

        # save_image(sar_log, f"/NAS_data/yjy/debug_sar_log.png")
        # save_image(opt_log, f"/NAS_data/yjy/debug_opt_log.png")
        # save_image(sar_raw, f"/NAS_data/yjy/debug_sar_raw.png")
        # save_image(opt_raw, f"/NAS_data/yjy/debug_opt_raw.png")
        return sar_log, opt_log


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

    sigmas = [1.0, 2.0]
    lap_ksize = 3
    clip_percentile = 99.5

    train_dataset = SAROptLoGDataset(
        sar_dir="/data/hjf/Dataset/SEN12_Scene/trainA/",
        opt_dir="/data/hjf/Dataset/SEN12_Scene/trainB/",
        img_size=256,
        sigmas=sigmas,
        lap_ksize=lap_ksize,
        clip_percentile=clip_percentile,
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=2, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 200
    save_dir = "/NAS_data/yjy/DDBM_GT_Unet/DDBM_S2O_LoG_results/SEN_scene"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for sar_log, opt_log in progress_bar:
            sar_log = sar_log.to(device, non_blocking=True)
            opt_log = opt_log.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(sar_log)
            loss = criterion(pred, opt_log)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sar_log, opt_log = next(iter(train_loader))
                sar_log = sar_log.to(device)
                opt_log = opt_log.to(device)
                pred = model(sar_log)

            sar_log, opt_log, pred = sar_log.cpu(), opt_log.cpu(), pred.cpu()
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            for i in range(3):
                axes[i, 0].imshow(sar_log[i][0], cmap='gray')
                axes[i, 0].set_title('Input SAR-LoG')
                axes[i, 1].imshow(pred[i][0].detach(), cmap='gray')
                axes[i, 1].set_title('Predicted LoG')
                axes[i, 2].imshow(opt_log[i][0], cmap='gray')
                axes[i, 2].set_title('GT Optical-LoG')
                for j in range(3):
                    axes[i, j].axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/epoch_{epoch + 1}.png")
            plt.close()

            torch.save(model.state_dict(), f"{save_dir}/unet_epoch_{epoch + 1}.pth")

    final_path = os.path.join(save_dir, "unet_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training finished! Final model saved to: {final_path}")
