import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn


# ------------------------------
# 你原始网络（可直接替换为项目中的同名类）
# ------------------------------
def canny_edge_detection(image, low_threshold=20, high_threshold=150):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim == 4:
        return np.stack([canny_edge_detection(img, low_threshold, high_threshold) for img in image])

    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return torch.from_numpy(edges).unsqueeze(0).repeat(3, 1, 1).float() / 255.0


def apply_canny_to_batch(batch_image):
    batch_size = batch_image.shape[0]
    canny_outputs = []
    for i in range(batch_size):
        canny_outputs.append(canny_edge_detection(batch_image[i]))
    return torch.stack(canny_outputs, dim=0)


class ConvNetworkWithImageFeature(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.ratio = 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.silu1 = nn.SiLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.silu2 = nn.SiLU()
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.silu3 = nn.SiLU()
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.conv_feature1 = nn.Conv2d(3, 64, kernel_size=1)
        self.conv_feature2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv_feature3 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_feature4 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, alpha):
        canny_output = apply_canny_to_batch(x).to(x.device)
        canny_resized = canny_output * alpha[0]

        x1 = self.silu1(self.conv1(x + canny_resized))

        canny_resized = self.conv_feature1(canny_resized)
        x2 = self.silu2(self.conv2(x1 + canny_resized * alpha[1]))

        canny_resized = self.conv_feature2(canny_resized)
        x3 = self.silu3(self.conv3(x2 + canny_resized * alpha[2]))

        canny_resized = self.conv_feature3(canny_resized)
        x4 = self.conv4(x3 + canny_resized * alpha[3]) * self.ratio
        x4 = x4 + x
        return x4


# ------------------------------
# 可视化与保存中间层输出
# ------------------------------
def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min, x_max = np.min(x), np.max(x)
    if np.isclose(x_max - x_min, 0):
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - x_min) / (x_max - x_min)
    return (x * 255).clip(0, 255).astype(np.uint8)


def save_feature_map_images(feat: torch.Tensor, out_dir: Path, layer_name: str, max_channels: int = 8):
    feat_np = feat.detach().cpu().numpy()  # [B, C, H, W]

    for b in range(feat_np.shape[0]):
        sample_dir = out_dir / f"sample_{b}" / layer_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 1) 平均通道图
        mean_map = np.mean(feat_np[b], axis=0)
        cv2.imwrite(str(sample_dir / "mean.png"), normalize_to_uint8(mean_map))

        # 2) 前 max_channels 个通道
        c = min(max_channels, feat_np[b].shape[0])
        for i in range(c):
            ch_map = feat_np[b, i]
            cv2.imwrite(str(sample_dir / f"ch_{i:03d}.png"), normalize_to_uint8(ch_map))


def register_hooks(model: nn.Module, layer_names: List[str], features: Dict[str, torch.Tensor]):
    hooks = []

    for name, module in model.named_modules():
        if name in layer_names:
            def _hook(m, i, o, key=name):
                features[key] = o
            hooks.append(module.register_forward_hook(_hook))

    return hooks

def load_user_images(image_path: str, in_channels: int, h: int, w: int, device: str) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if in_channels == 1:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif in_channels == 3:
        tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)  # [1,3,H,W]
    else:
        raise ValueError("Only in_channels=1 or in_channels=3 are supported by --image_path mode.")

    return tensor.to(device)



def main():
    parser = argparse.ArgumentParser(description="Save intermediate layer outputs as images.")
    parser.add_argument("--ckpt", type=str, default="/data/yjy_data/DDBM_GT_Unet/logs_SAR2OPT_0312_GT_se_SAB/ema_1_0.9999_220000.pt", help="checkpoint path (.pt/.pth)")
    parser.add_argument("--out_dir", type=str, default="layer_outputs_220000_1_3600_3840", help="output folder")
    parser.add_argument("--image_path", type=str, default="/NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA/1_3600_3840.jpg", help="your own input image path")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=512)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--max_channels", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConvNetworkWithImageFeature(args.in_channels, args.out_channels).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 你关心的层名：可按需增删
    target_layers = [
        "conv1", "silu1",
        "conv2", "silu2",
        "conv3", "silu3",
        "conv4",
        "conv_feature1", "conv_feature2", "conv_feature3", "conv_feature4",
    ]

    features: Dict[str, torch.Tensor] = {}
    hooks = register_hooks(model, target_layers, features)

    # 示例输入：真实使用时替换为你的 dataloader 一批图
    if args.image_path:
        x = load_user_images(args.image_path, args.in_channels, args.h, args.w, device)
    else:
        x = torch.rand(args.batch, args.in_channels, args.h, args.w, device=device)
    alpha = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)

    with torch.no_grad():
        out = model(x, alpha)
        features["final_output"] = out

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, feat in features.items():
        save_feature_map_images(feat, out_dir, name, max_channels=args.max_channels)

    for h in hooks:
        h.remove()

    print(f"Done. Saved layer maps to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()