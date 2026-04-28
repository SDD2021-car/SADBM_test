import os
import glob
from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# =========================
# 1) FAST corner heatmap
# =========================
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
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)

    if pre_blur:
        img_det = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        img_det = img

    fast = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=nonmax_suppression
    )
    kps = fast.detect(img_det, None)

    if top_k is not None and len(kps) > top_k:
        kps = sorted(kps, key=lambda k: k.response, reverse=True)[:top_k]

    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    if response_weighted and len(kps) > 0:
        rs = np.array([kp.response for kp in kps], dtype=np.float32)
        r_min, r_max = float(rs.min()), float(rs.max())
        denom = (r_max - r_min) if (r_max > r_min) else 1.0
    else:
        r_min, denom = 0.0, 1.0

    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= x < w and 0 <= y < h:
            if response_weighted:
                val = 0.5 + 0.5 * (float(kp.response) - r_min) / denom
            else:
                val = 1.0
            cv2.circle(heat, (x, y), radius, float(val), thickness=-1)

    if sigma is not None and sigma > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)

    m = float(heat.max())
    if m > 0:
        heat = heat / (m + 1e-8)

    return torch.from_numpy(heat).unsqueeze(0).float()  # [1,H,W]


# =========================
# 2) Dataset for test
# return: sar_corner, opt_corner, name
# =========================
class SAROptCornerTestDataset(Dataset):
    def __init__(self, sar_dir, opt_dir, img_size=256,
                 fast_threshold=50, fast_top_k=500):
        self.sar_files = sorted(glob.glob(os.path.join(sar_dir, "*.png")))
        self.opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.png")))
        assert len(self.sar_files) == len(self.opt_files), "数据数量不匹配"

        self.resize = transforms.Resize((img_size, img_size))
        self.fast_threshold = fast_threshold
        self.fast_top_k = fast_top_k

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar_path = self.sar_files[idx]
        opt_path = self.opt_files[idx]
        name = os.path.splitext(os.path.basename(sar_path))[0]

        sar = Image.open(sar_path).convert("L")
        opt = Image.open(opt_path).convert("L")

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

        return sar_corner.float(), opt_corner.float(), name


# =========================
# 3) UNet (same as train)
# =========================
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


# =========================
# 4) metrics (L1/MSE/PSNR/SSIM)
# =========================
def tensor_to_uint8_img(x: torch.Tensor) -> np.ndarray:
    """
    x: [1,H,W] or [H,W] in [0,1]
    return uint8 [H,W]
    """
    if x.ndim == 3:
        x = x[0]
    x = x.clamp(0, 1).detach().cpu().numpy()
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x

def calc_psnr(pred: np.ndarray, gt: np.ndarray, eps=1e-10) -> float:
    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0
    mse = np.mean((pred_f - gt_f) ** 2)
    if mse < eps:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))

def calc_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    # OpenCV SSIM for grayscale
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)

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


def save_gray_png(path: str, img_uint8: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_uint8)

def make_triplet_canvas(sar_u8: np.ndarray, pred_u8: np.ndarray, gt_u8: np.ndarray) -> np.ndarray:
    # [H,W] -> concat (SAR | Pred | GT)
    return np.concatenate([sar_u8, pred_u8, gt_u8], axis=1)


# =========================
# 5) main test
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- change these ----------
    test_sar_dir = "/data/hjf/Dataset/SEN12_Scene/testA/"
    test_opt_dir = "/data/hjf/Dataset/SEN12_Scene/testB/"
    model_path   = "/NAS_data/yjy/DDBM_GT_Unet/DDBM_S2O_FAST_corner_results/SEN_scene/unet_epoch_90.pth"
    save_dir     = "/NAS_data/yjy/DDBM_GT_Unet/DDBM_S2O_FAST_corner_results/SEN_scene_test_90"
    img_size     = 256
    fast_threshold = 50
    fast_top_k     = 500
    batch_size     = 8
    num_workers    = 2
    # --------------------------------

    os.makedirs(save_dir, exist_ok=True)
    out_sar_dir  = os.path.join(save_dir, "sar_corner")   # SAR提取结果
    out_gt_dir   = os.path.join(save_dir, "gt_corner")    # OPT提取结果 (GT)
    out_pred_dir = os.path.join(save_dir, "pred_corner")  # 模型输出 (Pred)
    out_vis_dir  = os.path.join(save_dir, "vis_triplet")  # 拼图可视化
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
    test_dataset = SAROptCornerTestDataset(
        sar_dir=test_sar_dir,
        opt_dir=test_opt_dir,
        img_size=img_size,
        fast_threshold=fast_threshold,
        fast_top_k=fast_top_k
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # metric accum
    rows = []
    for sar_corner, gt_corner, names in tqdm(test_loader, desc="Testing"):
        sar_corner = sar_corner.to(device, non_blocking=True)  # [B,1,H,W]
        gt_corner  = gt_corner.to(device, non_blocking=True)   # [B,1,H,W]

        with torch.no_grad():
            pred_corner = model(sar_corner)  # [B,1,H,W] in [0,1]

        # per-sample save + metric
        for b in range(sar_corner.size(0)):
            name = names[b]

            sar_u8  = tensor_to_uint8_img(sar_corner[b])
            gt_u8   = tensor_to_uint8_img(gt_corner[b])
            pred_u8 = tensor_to_uint8_img(pred_corner[b])

            # save images
            save_gray_png(os.path.join(out_sar_dir,  f"{name}.png"), sar_u8)
            save_gray_png(os.path.join(out_gt_dir,   f"{name}.png"), gt_u8)
            save_gray_png(os.path.join(out_pred_dir, f"{name}.png"), pred_u8)

            trip = make_triplet_canvas(sar_u8, pred_u8, gt_u8)
            save_gray_png(os.path.join(out_vis_dir, f"{name}.png"), trip)

            # metrics (compute on [0,1])
            pred_f = pred_u8.astype(np.float32) / 255.0
            gt_f   = gt_u8.astype(np.float32) / 255.0


    print("✅ Test finished!")
    print(f"Saved SAR corners : {out_sar_dir}")
    print(f"Saved GT corners  : {out_gt_dir}")
    print(f"Saved Pred corners: {out_pred_dir}")
    print(f"Saved Triplets    : {out_vis_dir}")
