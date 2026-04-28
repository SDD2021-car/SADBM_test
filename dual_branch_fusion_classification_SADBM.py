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


class PairedRSCDataset(Dataset):
    """
    root/
      Train/
        pred_pix2pix/ (optical, _s2_)
        A/            (sar,     _s1_)
      Test/
        pred_pix2pix/
        A/
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        optical_sub: str = "pred_SADBM",
        sar_sub: str = "A",
        resize: Tuple[int, int] = (256, 256),
    ):
        self.root_dir = Path(root_dir)
        self.opt_dir = self.root_dir / split / optical_sub
        self.sar_dir = self.root_dir / split / sar_sub
        self.is_train = split.lower() == "train"

        if not self.opt_dir.is_dir() or not self.sar_dir.is_dir():
            raise FileNotFoundError(f"Expect folders: {self.opt_dir} and {self.sar_dir}")

        opt_files = sorted(self.opt_dir.glob("*.png"))
        self.pairs = []
        for opt_path in opt_files:
            sar_name = opt_path.name.replace("_s2_", "_s1_")
            sar_path = self.sar_dir / sar_name
            if not sar_path.exists():
                raise FileNotFoundError(f"Pair for {opt_path.name} not found in {self.sar_dir}")
            self.pairs.append((sar_path, opt_path))

        self.sar_transform = T.Compose(
            [
                T.Grayscale(num_output_channels=1),
                T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        opt_aug = []
        if self.is_train:
            opt_aug = [
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomHorizontalFlip(p=0.5),
            ]

        self.opt_transform = T.Compose(
            opt_aug
            + [
                T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        sar_path, opt_path = self.pairs[idx]
        sar_img = Image.open(sar_path).convert("L")
        opt_img = Image.open(opt_path).convert("RGB")

        cls_name = parse_class_from_filename(opt_path.name)
        y_idx = NAME_TO_IDX[cls_name]

        return {
            "sar_image": self.sar_transform(sar_img),   # [1,H,W]
            "opt_image": self.opt_transform(opt_img),   # [3,H,W]
            "label_index": torch.tensor(y_idx, dtype=torch.long),
            "path_opt": str(opt_path),
            "path_sar": str(sar_path),
        }


def build_loader(root_dir: str, split: str, batch_size=32, num_workers=4, resize=(256, 256), shuffle=None):
    ds = PairedRSCDataset(root_dir=root_dir, split=split, resize=resize)
    if shuffle is None:
        shuffle = split.lower() == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        raise ValueError(f"in_ch_new={in_ch_new} < in_c_old={in_c_old}")

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


class GenericBackboneEncoder(nn.Module):
    def __init__(self, arch: str = "resnet18", in_ch: int = 3, ckpt_path: str = None):
        super().__init__()
        self.arch = arch
        self.is_resnet = arch in ["resnet18", "resnet34", "resnet50"]

        if self.is_resnet:
            if arch == "resnet18":
                base = models.resnet18(weights=None)
            elif arch == "resnet34":
                base = models.resnet34(weights=None)
            else:
                base = models.resnet50(weights=None)

            if in_ch != 3:
                old = base.conv1
                new = nn.Conv2d(
                    in_ch, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False
                )
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
                base.conv1 = new

            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.out_dim = base.fc.in_features
            self.img_size = None
        elif arch == "dino":
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=0,
                in_chans=in_ch,
            )
            img_size = self.backbone.patch_embed.img_size
            self.img_size = (img_size[0], img_size[1]) if isinstance(img_size, (tuple, list)) else (img_size, img_size)
            self.out_dim = self.backbone.num_features
            load_checkpoint_for_timm(self.backbone, ckpt_path, in_ch)
        elif arch == "dinov2":
            self.backbone = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=False,
                num_classes=0,
                in_chans=in_ch,
            )
            img_size = self.backbone.patch_embed.img_size
            self.img_size = (img_size[0], img_size[1]) if isinstance(img_size, (tuple, list)) else (img_size, img_size)
            self.out_dim = self.backbone.num_features
            load_checkpoint_for_timm(self.backbone, ckpt_path, in_ch)
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(self, x):
        if self.is_resnet:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            tokens = x.flatten(2).transpose(1, 2)  # [B,HW,C]
            return tokens

        h, w = x.shape[-2], x.shape[-1]
        if self.img_size is not None and (h, w) != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bicubic", align_corners=False)

        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict):
            feats = feats.get("x") or feats.get("pooled") or list(feats.values())[0]
        if feats.dim() == 2:
            feats = feats.unsqueeze(1)  # [B,1,C]
        return feats  # [B,N,C]


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.sar_to_opt = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.opt_to_sar = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln_sar = nn.LayerNorm(dim)
        self.ln_opt = nn.LayerNorm(dim)

    def forward(self, sar_tok, opt_tok):

        sar_msg, _ = self.sar_to_opt(query=sar_tok, key=opt_tok, value=opt_tok)
        opt_msg, _ = self.opt_to_sar(query=opt_tok, key=sar_tok, value=sar_tok)

        sar_out = self.ln_sar(sar_tok + sar_msg)
        opt_out = self.ln_opt(opt_tok + opt_msg)
        return sar_out, opt_out


class DualBranchHybridFusionNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        sar_arch: str = "resnet50",
        opt_arch: str = "resnet50",
        sar_ckpt: str = None,
        opt_ckpt: str = None,
        fusion_dim: int = 512,
    ):
        super().__init__()
        self.sar_encoder = GenericBackboneEncoder(arch=sar_arch, in_ch=1, ckpt_path=sar_ckpt)
        self.opt_encoder = GenericBackboneEncoder(arch=opt_arch, in_ch=3, ckpt_path=opt_ckpt)
        self.sar_proj = nn.Linear(self.sar_encoder.out_dim, fusion_dim)
        self.opt_proj = nn.Linear(self.opt_encoder.out_dim, fusion_dim)

        self.cross_fusion = CrossAttentionFusion(dim=fusion_dim, num_heads=8)

        # 特征级融合后的主头
        self.main_head = nn.Linear(fusion_dim * 2, num_classes)
        # 决策级融合的辅助头
        self.sar_aux_head = nn.Linear(fusion_dim, num_classes)
        self.opt_aux_head = nn.Linear(fusion_dim, num_classes)

    def forward(self, sar_x, opt_x):
        sar_tok = self.sar_proj(self.sar_encoder(sar_x))
        opt_tok = self.opt_proj(self.opt_encoder(opt_x))

        sar_tok, opt_tok = self.cross_fusion(sar_tok, opt_tok)
        sar_vec = sar_tok.mean(dim=1)
        opt_vec = opt_tok.mean(dim=1)

        main_logits = self.main_head(torch.cat([sar_vec, opt_vec], dim=1))
        sar_aux_logits = self.sar_aux_head(sar_vec)
        opt_aux_logits = self.opt_aux_head(opt_vec)

        # 决策级融合：主头 + 两个辅助头
        fused_logits = main_logits + 0.5 * sar_aux_logits + 0.5 * opt_aux_logits
        return {
            "fused_logits": fused_logits,
            "main_logits": main_logits,
            "sar_aux_logits": sar_aux_logits,
            "opt_aux_logits": opt_aux_logits,
        }


def hybrid_loss(outputs, targets, aux_weight=0.3):
    main = F.cross_entropy(outputs["main_logits"], targets)
    sar_aux = F.cross_entropy(outputs["sar_aux_logits"], targets)
    opt_aux = F.cross_entropy(outputs["opt_aux_logits"], targets)
    total = main + aux_weight * (sar_aux + opt_aux)
    return total, {"loss_main": main.item(), "loss_sar_aux": sar_aux.item(), "loss_opt_aux": opt_aux.item()}


def train_one_epoch(model, loader, optimizer, device, aux_weight=0.3):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        sar = batch["sar_image"].to(device, non_blocking=True)
        opt = batch["opt_image"].to(device, non_blocking=True)
        y = batch["label_index"].to(device, non_blocking=True)

        outputs = model(sar, opt)
        loss, _ = hybrid_loss(outputs, y, aux_weight=aux_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pred = outputs["fused_logits"].argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, aux_weight=0.3):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        sar = batch["sar_image"].to(device, non_blocking=True)
        opt = batch["opt_image"].to(device, non_blocking=True)
        y = batch["label_index"].to(device, non_blocking=True)

        outputs = model(sar, opt)
        loss, _ = hybrid_loss(outputs, y, aux_weight=aux_weight)
        pred = outputs["fused_logits"].argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def gather_logits_targets(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for batch in loader:
        sar = batch["sar_image"].to(device, non_blocking=True)
        opt = batch["opt_image"].to(device, non_blocking=True)
        y = batch["label_index"].to(device, non_blocking=True)
        outputs = model(sar, opt)
        all_logits.append(outputs["fused_logits"].detach().cpu())
        all_targets.append(y.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def compute_and_save_metrics(*, class_names, save_dir="metrics_out_dual", prefix="test", logits=None, targets=None):
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
    aa = pa.mean()
    f1_each = 2 * pa * ua / np.maximum(pa + ua, eps)
    f1_macro = np.nanmean(f1_each)
    weights = support / np.maximum(support.sum(), eps)
    f1_weighted = np.nansum(f1_each * weights)
    pe = (support * pred_sum).sum() / (n * n + eps)
    kappa = (oa - pe) / np.maximum(1 - pe, eps)
    top1 = (top1_pred == y_true).float().mean().item()
    top5 = (topk == y_true.unsqueeze(1)).any(dim=1).float().mean().item()

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
    cm_path = str(Path(save_dir) / f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)

    csv_path = str(Path(save_dir) / f"{prefix}_per_class_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "PA(Recall)", "UA(Precision)", "F1", "Support"])
        for i, name in enumerate(class_names):
            writer.writerow([name, f"{pa[i]:.6f}", f"{ua[i]:.6f}", f"{f1_each[i]:.6f}", int(support[i])])

    summary = {
        "N": int(n),
        "OA": float(oa),
        "AA": float(aa),
        "Kappa": float(kappa),
        "Top1": float(top1),
        "Top5": float(top5),
        "F1_macro": float(f1_macro),
        "F1_weighted": float(f1_weighted),
    }
    with open(Path(save_dir) / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/NAS_data/yjy/SEN1-2-BIT_matched")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--aux_weight", type=float, default=0.3, help="辅助损失权重")
    parser.add_argument("--backbone_arch", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "dino", "dinov2"])
    parser.add_argument("--sar_arch", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "dino", "dinov2"])
    parser.add_argument("--opt_arch", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "dino", "dinov2"])
    parser.add_argument("--dino_ckpt", type=str, default="/data/cyh/DOGAN/pretrained_dino/dino_vitbase16_pretrain.pth", help="DINO checkpoint path")
    parser.add_argument("--dinov2_ckpt", type=str, default="/NAS_data/hjf/DOGAN/DOGAN_resnet_DINOv2_ViTL14_resD/checkpoints/pretrained_dinov2/dinov2_vitb14_pretrain.pth", help="DINOv2 checkpoint path")
    parser.add_argument("--sar_ckpt", type=str, default=None, help="SAR分支checkpoint路径（优先级最高）")
    parser.add_argument("--opt_ckpt", type=str, default=None, help="光学分支checkpoint路径（优先级最高）")
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="固定随机种子")
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = build_loader(args.root, "Train", batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = build_loader(
        args.root, "Test", batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    sar_arch = args.sar_arch or args.backbone_arch
    opt_arch = args.opt_arch or args.backbone_arch

    def pick_ckpt(arch_name: str, manual_path: str):
        if manual_path:
            return manual_path
        if arch_name == "dino":
            return args.dino_ckpt
        if arch_name == "dinov2":
            return args.dinov2_ckpt
        return None

    sar_ckpt = pick_ckpt(sar_arch, args.sar_ckpt)
    opt_ckpt = pick_ckpt(opt_arch, args.opt_ckpt)
    print(f"Backbone config -> SAR: {sar_arch}, OPT: {opt_arch}")

    model = DualBranchHybridFusionNet(
        num_classes=len(CLASS_NAMES),
        sar_arch=sar_arch,
        opt_arch=opt_arch,
        sar_ckpt=sar_ckpt,
        opt_ckpt=opt_ckpt,
        fusion_dim=args.fusion_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs("classification_checkpoints", exist_ok=True)
    os.makedirs("metrics_dual_branch", exist_ok=True)
    best_acc = 0.0
    best_path = "classification_checkpoints/best_dual_branch_fusion_SADBM_res50.pth"

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, aux_weight=args.aux_weight)
        te_loss, te_acc = evaluate(model, test_loader, device, aux_weight=args.aux_weight)
        print(
            f"[DualBranch] Epoch {ep:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"test loss {te_loss:.4f} acc {te_acc:.4f}"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best at epoch {ep}, acc={best_acc:.4f} -> {best_path}")

        logits, targets = gather_logits_targets(model, test_loader, device)
        compute_and_save_metrics(
            class_names=CLASS_NAMES,
            save_dir="metrics_dual_branch_SADBM_res50",
            prefix=f"dual_branch_epoch{ep:03d}_test",
            logits=logits,
            targets=targets,
        )

    print(f"✅ Finished training. Best test acc={best_acc:.4f}")


if __name__ == "__main__":
    main()