import os
import glob
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
# 1️⃣ Canny 边缘提取（纯 OpenCV Canny）
# ============================================================
def canny_edge_from_pil(
    pil_img: Image.Image,
    low_th: int = 50,
    high_th: int = 150,
    blur_ksize: int = 3,         # 0 表示不做 blur
    aperture_size: int = 3,      # Sobel kernel size inside Canny
    L2gradient: bool = True
) -> torch.Tensor:
    """
    输入: PIL RGB/Gray
    输出: [1,H,W] float Tensor in [0,1]
    """
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)

    if blur_ksize and blur_ksize > 0:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    edge = cv2.Canny(
        image=img,
        threshold1=low_th,
        threshold2=high_th,
        apertureSize=aperture_size,
        L2gradient=L2gradient
    )  # uint8, {0,255}

    edge = edge.astype(np.float32) / 255.0  # -> [0,1]
    return torch.from_numpy(edge).unsqueeze(0)  # [1,H,W]


# ============================================================
# 2️⃣ Dataset：直接返回 SAR_canny, OPT_canny
# ============================================================
class SAROptCannyDataset(Dataset):
    def __init__(self, sar_dir, opt_dir, img_size=256,
                 low_th=50, high_th=150, blur_ksize=3,
                 debug_dir=None, debug_first_n=0):
        self.sar_files = sorted(glob.glob(os.path.join(sar_dir, "*.png")))
        self.opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.png")))
        assert len(self.sar_files) == len(self.opt_files), "数据数量不匹配"

        self.resize = transforms.Resize((img_size, img_size))
        self.low_th = low_th
        self.high_th = high_th
        self.blur_ksize = blur_ksize

        self.debug_dir = debug_dir
        self.debug_first_n = debug_first_n
        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar = Image.open(self.sar_files[idx]).convert("RGB")
        opt = Image.open(self.opt_files[idx]).convert("RGB")

        sar = self.resize(sar)
        opt = self.resize(opt)

        sar_edge = canny_edge_from_pil(
            sar, low_th=self.low_th, high_th=self.high_th, blur_ksize=self.blur_ksize
        )
        opt_edge = canny_edge_from_pil(
            opt, low_th=self.low_th, high_th=self.high_th, blur_ksize=self.blur_ksize
        )

        # 可选：只保存前 N 张 debug（避免写太多、拖慢）
        if self.debug_dir is not None and idx < self.debug_first_n:
            save_image(sar_edge, os.path.join(self.debug_dir, f"sar_canny_{idx:06d}.png"))
            save_image(opt_edge, os.path.join(self.debug_dir, f"opt_canny_{idx:06d}.png"))

        return sar_edge.float(), opt_edge.float()


# ============================================================
# 3️⃣ U-Net 模型定义（不变）
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

    # Canny 参数（你可以做子实验：调阈值/blur）
    low_th = 50
    high_th = 150
    blur_ksize = 3  # SAR speckle 多时建议 3 或 5

    train_dataset = SAROptCannyDataset(
        sar_dir="/data/hjf/Dataset/SEN12_Scene/trainA/",
        opt_dir="/data/hjf/Dataset/SEN12_Scene/trainB/",
        img_size=256,
        low_th=low_th,
        high_th=high_th,
        blur_ksize=blur_ksize,
        debug_dir="/NAS_data/yjy/debug_canny",   # 不想保存就设 None
        debug_first_n=10
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=2, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 200
    save_dir = "/NAS_data/yjy/DDBM_GT_Unet/DDBM_S2O_Canny_Original_results"
    os.makedirs(save_dir, exist_ok=True)

    # ============================================================
    # 5️⃣ 训练循环
    # ============================================================
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for sar_edge, opt_edge in progress_bar:
            sar_edge = sar_edge.to(device, non_blocking=True)
            opt_edge = opt_edge.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_edge = model(sar_edge)
            loss = criterion(pred_edge, opt_edge)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sar_edge, opt_edge = next(iter(train_loader))
                sar_edge = sar_edge.to(device)
                opt_edge = opt_edge.to(device)
                pred = model(sar_edge)

            sar_edge, opt_edge, pred = sar_edge.cpu(), opt_edge.cpu(), pred.cpu()
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            for i in range(3):
                axes[i, 0].imshow(sar_edge[i][0], cmap='gray')
                axes[i, 0].set_title('Input SAR-Canny')
                axes[i, 1].imshow(pred[i][0].detach(), cmap='gray')
                axes[i, 1].set_title('Predicted Canny')
                axes[i, 2].imshow(opt_edge[i][0], cmap='gray')
                axes[i, 2].set_title('GT Optical-Canny')
                for j in range(3):
                    axes[i, j].axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/epoch_{epoch + 1}.png")
            plt.close()

            torch.save(model.state_dict(), f"{save_dir}/unet_epoch_{epoch + 1}.pth")

    # ============================================================
    # 6️⃣ 保存最终模型
    # ============================================================
    final_path = os.path.join(save_dir, "unet_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training finished! Final model saved to: {final_path}")
