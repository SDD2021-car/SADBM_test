
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
# 1️⃣ FAST 角点提取：把关键点渲染成 1 通道 corner map
# ============================================================
def fast_corner_heatmap_from_pil(
    pil_img: Image.Image,
    threshold: int = 40,
    nonmax_suppression: bool = True,
    top_k: Optional[int] = 600,
    radius: int = 3,
    sigma: float = 1.6,
    response_weighted: bool = True,
    pre_blur: bool = True,
) -> torch.Tensor:
    """
    输入: PIL图像(灰度 or RGB)
    输出: [1,H,W] float32 Tensor, 范围[0,1]，连续角点热力图
    """
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)

    # 预平滑：抑制噪声/散斑导致的伪角点（建议保留）
    if pre_blur:
        img_det = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        img_det = img

    fast = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=nonmax_suppression
    )
    kps = fast.detect(img_det, None)

    # 限制角点数量：避免“撒盐”
    if top_k is not None and len(kps) > top_k:
        kps = sorted(kps, key=lambda k: k.response, reverse=True)[:top_k]

    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    # 将 kp.response 归一化到 [0.5,1] 左右，用于加权（可选）
    if response_weighted and len(kps) > 0:
        rs = np.array([kp.response for kp in kps], dtype=np.float32)
        r_min, r_max = float(rs.min()), float(rs.max())
        denom = (r_max - r_min) if (r_max > r_min) else 1.0
    else:
        r_min, denom = 0.0, 1.0

    # 渲染：每个角点画一个小圆（或点），强度可按 response 加权
    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= x < w and 0 <= y < h:
            if response_weighted:
                val = 0.5 + 0.5 * (float(kp.response) - r_min) / denom  # [0.5,1]
            else:
                val = 1.0
            cv2.circle(heat, (x, y), radius, float(val), thickness=-1)

    # 高斯平滑：把“点”变成连续热力图（关键）
    if sigma is not None and sigma > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 归一化到 [0,1]
    m = float(heat.max())
    if m > 0:
        heat = heat / (m + 1e-8)

    return torch.from_numpy(heat).unsqueeze(0).float()  # [1,H,W]



# ============================================================
# 2️⃣ Dataset：直接返回 SAR_corner, OPT_corner
# ============================================================
class SAROptCornerDataset(Dataset):
    def __init__(self, sar_dir, opt_dir, img_size=256,
                 fast_threshold=20, fast_top_k=800):
        self.sar_files = sorted(glob.glob(os.path.join(sar_dir, "*.png")))
        self.opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.png")))
        assert len(self.sar_files) == len(self.opt_files), "数据数量不匹配"

        self.resize = transforms.Resize((img_size, img_size))
        self.fast_threshold = fast_threshold
        self.fast_top_k = fast_top_k

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar = Image.open(self.sar_files[idx]).convert("L")
        opt = Image.open(self.opt_files[idx]).convert("L")

        sar = self.resize(sar)
        opt = self.resize(opt)

        sar_corner = fast_corner_heatmap_from_pil(
            sar,
            threshold=self.fast_threshold,
            top_k=self.fast_top_k,
            radius=4,
            sigma=1.6,
            response_weighted=True,
            pre_blur=True,
        )
        opt_corner = fast_corner_heatmap_from_pil(
            opt,
            threshold=self.fast_threshold,
            top_k=self.fast_top_k,
            radius=4,
            sigma=1.6,
            response_weighted=True,
            pre_blur=True,
        )

        # save_image(sar_corner, "/NAS_data/yjy/sar.png")
        # save_image(opt_corner, "/NAS_data/yjy/opt.png")
        return sar_corner.float(), opt_corner.float()


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

    # 你可以重点调这两个参数：
    # - threshold：越大角点越少（SAR speckle 多时可以适当调大）
    # - top_k：限制角点数量（让训练更稳定）
    fast_threshold = 50
    fast_top_k = 500

    train_dataset = SAROptCornerDataset(
        sar_dir="/data/hjf/Dataset/SEN12_Scene/trainA/",
        opt_dir="/data/hjf/Dataset/SEN12_Scene/trainB/",
        img_size=256,
        fast_threshold=fast_threshold,
        fast_top_k=fast_top_k
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=2, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 200
    save_dir = "/NAS_data/yjy/DDBM_GT_Unet/DDBM_S2O_FAST_corner_results/SEN_scene"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for sar_corner, opt_corner in progress_bar:
            sar_corner = sar_corner.to(device, non_blocking=True)
            opt_corner = opt_corner.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_corner = model(sar_corner)
            loss = criterion(pred_corner, opt_corner)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sar_corner, opt_corner = next(iter(train_loader))
                sar_corner = sar_corner.to(device)
                opt_corner = opt_corner.to(device)
                pred = model(sar_corner)

            sar_corner, opt_corner, pred = sar_corner.cpu(), opt_corner.cpu(), pred.cpu()
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            for i in range(3):
                axes[i, 0].imshow(sar_corner[i][0], cmap='gray')
                axes[i, 0].set_title('Input SAR-Corner')
                axes[i, 1].imshow(pred[i][0].detach(), cmap='gray')
                axes[i, 1].set_title('Predicted Corner')
                axes[i, 2].imshow(opt_corner[i][0], cmap='gray')
                axes[i, 2].set_title('GT Optical-Corner')
                for j in range(3):
                    axes[i, j].axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/epoch_{epoch + 1}.png")
            plt.close()

            torch.save(model.state_dict(), f"{save_dir}/unet_epoch_{epoch + 1}.pth")

    final_path = os.path.join(save_dir, "unet_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training finished! Final model saved to: {final_path}")
