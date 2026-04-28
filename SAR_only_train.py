import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import timm
import random

CLASS_NAMES = ["Farmland", "Forest", "Mountain", "Rural", "Semi-Urban", "Urban"]
NAME_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def parse_class_from_filename(fname: str) -> str:
    stem = Path(fname).stem
    cls = stem.split("_")[-1]
    if cls not in NAME_TO_IDX:
        raise ValueError(f"Unrecognized class in filename: {fname}")
    return cls


class SAROnlyDataset(Dataset):
    """
    root/
      Train/
        A/ (sar, _s1_)
      Test/
        A/
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        sar_sub: str = "A",
        resize: Tuple[int, int] = (256, 256),
    ):
        self.root_dir = Path(root_dir)
        self.sar_dir = self.root_dir / split / sar_sub
        if not self.sar_dir.is_dir():
            raise FileNotFoundError(f"Expect folder: {self.sar_dir}")

        self.sar_files = sorted(self.sar_dir.glob("*.png"))
        if len(self.sar_files) == 0:
            raise FileNotFoundError(f"No png files found in {self.sar_dir}")

        self.sar_transform = T.Compose(
            [
                T.Grayscale(num_output_channels=1),
                T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx: int) -> Dict:
        sar_path = self.sar_files[idx]
        sar_img = Image.open(sar_path).convert("L")
        cls_name = parse_class_from_filename(sar_path.name)
        y_idx = NAME_TO_IDX[cls_name]
        return {
            "sar_image": self.sar_transform(sar_img),
            "label_index": torch.tensor(y_idx, dtype=torch.long),
            "path_sar": str(sar_path),
        }


def build_loader(root_dir: str, split: str, batch_size=32, num_workers=4, resize=(256, 256), shuffle=None):
    ds = SAROnlyDataset(root_dir=root_dir, split=split, resize=resize)
    if shuffle is None:
        shuffle = split.lower() == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def expand_weight_in_state_dict(state_dict: dict, key: str, in_ch_new: int):
    if key not in state_dict:
        return state_dict
    w = state_dict[key]
    if w.dim() != 4:
        return state_dict
    out_c, in_c_old, k_h, k_w = w.shape
    if in_c_old == in_ch_new:
        return state_dict
    if in_ch_new < in_c_old:
        # 3->1 这种情况取均值压缩
        state_dict[key] = w.mean(dim=1, keepdim=True)
        return state_dict

    new_w = w.new_zeros((out_c, in_ch_new, k_h, k_w))
    new_w[:, :in_c_old] = w
    extra = in_ch_new - in_c_old
    mean_w = w.mean(dim=1, keepdim=True)
    new_w[:, in_c_old:] = mean_w.repeat(1, extra, 1, 1)
    state_dict[key] = new_w
    return state_dict


def load_checkpoint_for_timm(backbone: nn.Module, ckpt_path: str, in_ch: int):
    if not ckpt_path:
        return
    print(f"[Backbone] Load checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict):
        inner = state_dict.get("model") or state_dict.get("state_dict") or state_dict
    else:
        inner = state_dict

    new_state = {}
    for k, v in inner.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_state[nk] = v
    new_state = expand_weight_in_state_dict(new_state, "patch_embed.proj.weight", in_ch_new=in_ch)
    msg = backbone.load_state_dict(new_state, strict=False)
    print("[Backbone] load_state_dict message:", msg)


class SARBackboneClassifier(nn.Module):
    def __init__(self, arch: str = "resnet18", num_classes: int = 6, in_ch: int = 1, ckpt_path: str = None):
        super().__init__()
        self.arch = arch
        self.is_resnet = arch in ["resnet18", "resnet34", "resnet50"]

        if self.is_resnet:
            if arch == "resnet18":
                model = models.resnet18(weights=None)
            elif arch == "resnet34":
                model = models.resnet34(weights=None)
            else:
                model = models.resnet50(weights=None)

            if in_ch != 3:
                old = model.conv1
                model.conv1 = nn.Conv2d(
                    in_ch, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False
                )
                nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            self.backbone = model
            self.img_size = None
        elif arch == "dino":
            self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, in_chans=in_ch)
            img_size = self.backbone.patch_embed.img_size
            self.img_size = (img_size[0], img_size[1]) if isinstance(img_size, (tuple, list)) else (img_size, img_size)
            self.head = nn.Linear(self.backbone.num_features, num_classes)
            load_checkpoint_for_timm(self.backbone, ckpt_path, in_ch)
        elif arch == "dinov2":
            self.backbone = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, in_chans=in_ch
            )
            img_size = self.backbone.patch_embed.img_size
            self.img_size = (img_size[0], img_size[1]) if isinstance(img_size, (tuple, list)) else (img_size, img_size)
            self.head = nn.Linear(self.backbone.num_features, num_classes)
            load_checkpoint_for_timm(self.backbone, ckpt_path, in_ch)
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(self, x):
        if self.is_resnet:
            return self.backbone(x)

        h, w = x.shape[-2], x.shape[-1]
        if self.img_size is not None and (h, w) != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bicubic", align_corners=False)
        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict):
            feats = feats.get("x") or feats.get("pooled") or list(feats.values())[0]
        if feats.dim() == 3:
            feats = feats.mean(dim=1)
        return self.head(feats)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        x = batch["sar_image"].to(device, non_blocking=True)
        y = batch["label_index"].to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        x = batch["sar_image"].to(device, non_blocking=True)
        y = batch["label_index"].to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def gather_logits_targets(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for batch in loader:
        x = batch["sar_image"].to(device, non_blocking=True)
        y = batch["label_index"].to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def compute_and_save_metrics(*, class_names, save_dir="metrics_out", prefix="test", logits=None, targets=None):
    os.makedirs(save_dir, exist_ok=True)
    c = len(class_names)
    top1_pred = torch.argmax(logits, dim=1).cpu()
    k = min(5, logits.shape[1])
    topk = torch.topk(logits, k=k, dim=1).indices.cpu()
    y_true = targets.cpu()
    n = y_true.numel()

    conf = torch.zeros((c, c), dtype=torch.int64)
    for t, p in zip(y_true, top1_pred):
        conf[t, p] += 1
    conf_np = conf.numpy()

    oa = conf.trace().item() / n
    support = conf.sum(axis=1).numpy()
    pred_sum = conf.sum(axis=0).numpy()
    tp = np.diag(conf_np).astype(np.float64)
    eps = 1e-12
    pa = tp / np.maximum(support, eps)
    ua = tp / np.maximum(pred_sum, eps)
    f1_each = 2 * pa * ua / np.maximum(pa + ua, eps)
    summary = {
        "N": int(n),
        "OA": float(oa),
        "AA": float(pa.mean()),
        "Kappa": float((oa - (support * pred_sum).sum() / (n * n + eps)) / np.maximum(1 - (support * pred_sum).sum() / (n * n + eps), eps)),
        "Top1": float((top1_pred == y_true).float().mean().item()),
        "Top5": float((topk == y_true.unsqueeze(1)).any(dim=1).float().mean().item()),
        "F1_macro": float(np.nanmean(f1_each)),
        "F1_weighted": float(np.nansum(f1_each * (support / np.maximum(support.sum(), eps)))),
    }

    with open(Path(save_dir) / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(Path(save_dir) / f"{prefix}_confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(class_names))
        for i, name in enumerate(class_names):
            writer.writerow([name] + list(map(int, conf_np[i])))

    fig = plt.figure(figsize=(6.2, 5.5), dpi=160)
    ax = plt.gca()
    im = ax.imshow(conf_np, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(c))
    ax.set_yticks(range(c))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{prefix}_confusion_matrix.png")
    plt.close(fig)
    return summary


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/NAS_data/yjy/SEN1-2-BIT_matched")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50", "dino", "dinov2"])
    parser.add_argument("--dino_ckpt", type=str, default="/data/cyh/DOGAN/pretrained_dino/dino_vitbase16_pretrain.pth")
    parser.add_argument("--dinov2_ckpt", type=str, default="/NAS_data/hjf/DOGAN/DOGAN_resnet_DINOv2_ViTL14_resD/checkpoints/pretrained_dinov2/dinov2_vitb14_pretrain.pth")
    parser.add_argument("--ckpt_path", type=str, default=None, help="手动指定 backbone ckpt，优先级最高")
    parser.add_argument("--seed", type=int, default=42, help="固定随机种子")
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = build_loader(args.root, "Train", batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = build_loader(args.root, "Test", batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    ckpt = args.ckpt_path
    if ckpt is None:
        ckpt = args.dino_ckpt if args.arch == "dino" else args.dinov2_ckpt if args.arch == "dinov2" else None

    model = SARBackboneClassifier(arch=args.arch, num_classes=len(CLASS_NAMES), in_ch=1, ckpt_path=ckpt).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs("classification_checkpoints", exist_ok=True)
    os.makedirs("metrics_sar_only", exist_ok=True)
    best_acc = 0.0
    best_path = f"classification_checkpoints/best_sar_only_{args.arch}_res50.pth"

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"[SAR-Only|{args.arch}] Epoch {ep:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | test {te_loss:.4f}/{te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best at epoch {ep}, acc={best_acc:.4f} -> {best_path}")

        logits, targets = gather_logits_targets(model, test_loader, device)
        compute_and_save_metrics(
            class_names=CLASS_NAMES,
            save_dir="metrics_sar_only_res50",
            prefix=f"sar_only_{args.arch}_epoch{ep:03d}_test",
            logits=logits,
            targets=targets,
        )

    print(f"✅ Finished SAR-only training. Best test acc={best_acc:.4f}")


if __name__ == "__main__":
    main()