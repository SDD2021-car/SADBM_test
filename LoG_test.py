import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ============================================================
# 1️⃣ LoG 特征提取：Gaussian + Laplacian (Laplacian of Gaussian)
# ============================================================
def log_response_map_from_pil(
    pil_img: Image.Image,
    sigmas: List[float] = [1.0, 2.0],
    lap_ksize: int = 3,
    pre_blur: bool = False,
    pre_blur_ksize: int = 3,
    take_abs: bool = True,
    clip_percentile: float = 99.5,
    normalize: bool = True,
) -> torch.Tensor:
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

    return torch.from_numpy(R).unsqueeze(0).float()  # [1,H,W]


# ============================================================
# 2️⃣ Test Dataset：返回 SAR_LoG, OPT_LoG, name
# ============================================================
class SAROptLoGTestDataset(Dataset):
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
        sar_path = self.sar_files[idx]
        opt_path = self.opt_files[idx]
        name = os.path.splitext(os.path.basename(sar_path))[0]

        sar = Image.open(sar_path).convert("L")
        opt = Image.open(opt_path).convert("RGB")

        sar = self.resize(sar)
        opt = self.resize(opt)

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

        return sar_log, opt_log, name


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
# 4️⃣ 保存与指标
# ============================================================
def tensor01_to_uint8(x: torch.Tensor) -> np.ndarray:
    """
    x: [1,H,W] or [H,W], value in [0,1]
    return uint8 [H,W]
    """
    if x.ndim == 3:
        x = x[0]
    x = x.clamp(0, 1).detach().cpu().numpy()
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x

def save_gray_png(path: str, img_u8: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_u8)

def calc_psnr_u8(pred_u8: np.ndarray, gt_u8: np.ndarray, eps=1e-10) -> float:
    pred = pred_u8.astype(np.float32) / 255.0
    gt = gt_u8.astype(np.float32) / 255.0
    mse = float(np.mean((pred - gt) ** 2))
    if mse < eps:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))

def calc_ssim_u8(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    pred_f = pred_u8.astype(np.float32)
    gt_f = gt_u8.astype(np.float32)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(pred_f, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(gt_f, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(pred_f * pred_f, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(gt_f * gt_f, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(pred_f * gt_f, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())

def make_triplet(sar_u8: np.ndarray, pred_u8: np.ndarray, gt_u8: np.ndarray) -> np.ndarray:
    return np.concatenate([sar_u8, pred_u8, gt_u8], axis=1)


# ============================================================
# 5️⃣ Test main
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 改这里 ==========
    test_sar_dir = "/data/hjf/Dataset/SEN12_Scene/testA/"
    test_opt_dir = "/data/hjf/Dataset/SEN12_Scene/testB/"

    model_path = "/NAS_data/yjy/DDBM_GT_Unet/DDBM_S2O_LoG_results/SEN_scene/unet_final.pth"
    save_dir   = "/NAS_data/yjy/DDBM_GT_Unet/DDBM_S2O_LoG_results/SEN_scene_test"

    img_size = 256

    # LoG 参数：建议与训练保持一致
    sigmas = [1.0, 2.0]
    lap_ksize = 3
    clip_percentile = 99.5

    batch_size = 8
    num_workers = 2
    # ===========================

    os.makedirs(save_dir, exist_ok=True)
    out_sar_dir  = os.path.join(save_dir, "sar_log")     # SAR 提取结果
    out_gt_dir   = os.path.join(save_dir, "gt_log")      # OPT 提取结果 (GT)
    out_pred_dir = os.path.join(save_dir, "pred_log")    # 模型输出 (Pred)
    out_vis_dir  = os.path.join(save_dir, "vis_triplet") # 拼图
    os.makedirs(out_sar_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)
    os.makedirs(out_pred_dir, exist_ok=True)
    os.makedirs(out_vis_dir, exist_ok=True)

    # load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    # dataloader
    test_dataset = SAROptLoGTestDataset(
        sar_dir=test_sar_dir,
        opt_dir=test_opt_dir,
        img_size=img_size,
        sigmas=sigmas,
        lap_ksize=lap_ksize,
        clip_percentile=clip_percentile,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    for sar_log, gt_log, names in tqdm(test_loader, desc="Testing(LoG)"):
        sar_log = sar_log.to(device, non_blocking=True)  # [B,1,H,W]
        gt_log  = gt_log.to(device, non_blocking=True)   # [B,1,H,W]

        with torch.no_grad():
            pred_log = model(sar_log)  # [B,1,H,W] in [0,1]

        for b in range(sar_log.size(0)):
            name = names[b]

            sar_u8  = tensor01_to_uint8(sar_log[b])
            gt_u8   = tensor01_to_uint8(gt_log[b])
            pred_u8 = tensor01_to_uint8(pred_log[b])

            # 保存三类图像
            save_gray_png(os.path.join(out_sar_dir,  f"{name}.png"), sar_u8)
            save_gray_png(os.path.join(out_gt_dir,   f"{name}.png"), gt_u8)
            save_gray_png(os.path.join(out_pred_dir, f"{name}.png"), pred_u8)

            # 拼图：SAR | Pred | GT
            trip = make_triplet(sar_u8, pred_u8, gt_u8)
            save_gray_png(os.path.join(out_vis_dir, f"{name}.png"), trip)

            # metrics (on [0,1])
            pred_f = pred_u8.astype(np.float32) / 255.0
            gt_f   = gt_u8.astype(np.float32) / 255.0


    print("✅ LoG Test finished!")
    print(f"Saved SAR LoG  : {out_sar_dir}")
    print(f"Saved GT  LoG  : {out_gt_dir}")
    print(f"Saved Pred LoG : {out_pred_dir}")
    print(f"Saved Triplets : {out_vis_dir}")
