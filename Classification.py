import os
from pathlib import Path
from typing import Tuple, Dict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torch.optim import AdamW
import argparse


import json, csv, os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def gather_logits_targets(model, loader, device):
    """
    额外的一次前向：收集全量 logits 和 targets（不改原 evaluate）。
    返回：
      logits: [N, C] 的 torch.Tensor
      targets: [N] 的 torch.LongTensor
    """
    model.eval()
    all_logits, all_targets = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label_index"].to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def compute_and_save_metrics(
    *,
    class_names,
    save_dir="metrics_out",
    prefix="test",
    logits: torch.Tensor = None,
    preds: torch.Tensor = None,
    targets: torch.Tensor,
):
    """
    计算并保存：Top-1/Top-5、OA、CA/PA、UA、AA、F1(macro/weighted)、Kappa、Confusion Matrix(图/CSV/JSON)
    使用方式（推荐）：
        logits, targets = gather_logits_targets(model, loader, device)
        compute_and_save_metrics(class_names=CLASS_NAMES, save_dir="metrics",
                                 prefix="epoch30_test", logits=logits, targets=targets)

    仅有 preds 时也可：
        compute_and_save_metrics(class_names=..., preds=preds, targets=targets)
      （此时无法算 Top-5，会置为 NaN）
    """
    os.makedirs(save_dir, exist_ok=True)
    C = len(class_names)

    # ===== 准备预测 =====
    if logits is None and preds is None:
        raise ValueError("Provide either logits or preds.")
    if logits is not None:
        # Top-1
        top1_pred = torch.argmax(logits, dim=1).cpu()
        # Top-5（若类别<5，则用 min(5,C)）
        k = min(5, logits.shape[1])
        topk = torch.topk(logits, k=k, dim=1).indices.cpu()  # [N, k]
    else:
        top1_pred = preds.cpu()
        topk = None  # 无法计算 top-5

    y_true = targets.cpu()
    N = y_true.numel()

    # ===== Confusion Matrix =====
    conf = torch.zeros((C, C), dtype=torch.int64)  # [true, pred]
    for t, p in zip(y_true, top1_pred):
        conf[t, p] += 1
    conf_np = conf.numpy()

    # ===== 指标计算 =====
    # OA
    OA = conf.trace().item() / N

    # 每类支持数
    support = conf.sum(axis=1).numpy()             # 每个类别的真实数量（行和）
    pred_sum = conf.sum(axis=0).numpy()            # 每个类别的预测数量（列和）
    TP = np.diag(conf_np).astype(np.float64)

    eps = 1e-12
    # CA/PA（召回 Recall）：TP / row_sum
    PA = TP / np.maximum(support, eps)
    # UA（精确率 Precision）：TP / col_sum
    UA = TP / np.maximum(pred_sum, eps)
    # AA：平均召回
    AA = PA.mean()

    # F1 per-class
    F1_per_class = 2 * PA * UA / np.maximum(PA + UA, eps)
    F1_macro = np.nanmean(F1_per_class)
    # weighted F1：按支持数加权
    weights = support / np.maximum(support.sum(), eps)
    F1_weighted = np.nansum(F1_per_class * weights)

    # Kappa
    po = OA
    pe = (support * pred_sum).sum() / (N * N + eps)
    kappa = (po - pe) / np.maximum(1 - pe, eps)

    # Top-1 / Top-5
    top1_acc = (top1_pred == y_true).float().mean().item()
    if topk is not None:
        in_topk = (topk == y_true.unsqueeze(1)).any(dim=1).float().mean().item()
        top5_acc = in_topk
    else:
        top5_acc = float("nan")

    # ===== 保存混淆矩阵图 =====
    fig = plt.figure(figsize=(6.2, 5.5), dpi=160)
    ax = plt.gca()
    im = ax.imshow(conf_np, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    # 数字标注（可选，样本多时可注释掉）
    for i in range(C):
        for j in range(C):
            v = conf_np[i, j]
            if v > 0:
                ax.text(j, i, str(v), va="center", ha="center", fontsize=8)
    plt.tight_layout()
    cm_path = str(Path(save_dir) / f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)

    # ===== 保存 CSV（每类 UA/PA/F1/support）=====
    csv_path = str(Path(save_dir) / f"{prefix}_per_class_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "PA(Recall)", "UA(Precision)", "F1", "Support"])
        for i, name in enumerate(class_names):
            writer.writerow([name, f"{PA[i]:.6f}", f"{UA[i]:.6f}", f"{F1_per_class[i]:.6f}", int(support[i])])

    # ===== 保存总体指标 JSON =====
    summary = {
        "N": int(N),
        "OA": float(OA),
        "AA": float(AA),
        "Kappa": float(kappa),
        "Top1": float(top1_acc),
        "Top5": float(top5_acc),
        "F1_macro": float(F1_macro),
        "F1_weighted": float(F1_weighted),
    }
    json_path = str(Path(save_dir) / f"{prefix}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ===== 保存混淆矩阵 CSV（可用于Latex或后处理）=====
    cm_csv_path = str(Path(save_dir) / f"{prefix}_confusion_matrix.csv")
    with open(cm_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(class_names))
        for i, name in enumerate(class_names):
            writer.writerow([name] + list(map(int, conf_np[i])))

    print(f"[Metrics] Saved to: {save_dir}")
    print(f"  Summary JSON : {json_path}")
    print(f"  Per-class CSV: {csv_path}")
    print(f"  ConfMat PNG  : {cm_path}")
    print(f"  ConfMat CSV  : {cm_csv_path}")
    return summary



# ====== 数据集配置 ======
CLASS_NAMES = ["Farmland", "Forest", "Mountain", "Rural", "Semi-Urban", "Urban"]
NAME_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def parse_class_from_filename(fname: str) -> str:
    stem = Path(fname).stem
    cls = stem.split('_')[-1]
    if cls not in NAME_TO_IDX:
        raise ValueError(f"Unrecognized class in filename: {fname}")
    return cls

def make_one_hot(idx: int, num_classes: int) -> torch.Tensor:
    v = torch.zeros(num_classes, dtype=torch.float32)
    v[idx] = 1.0
    return v


class PairedRSCDataset(Dataset):
    """
    目录结构：
      root/
        Train/
          o/  (optical, 文件名含 s2)
          s/  (SAR,     文件名含 s1)
        Test/
          o/
          s/
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        optical_sub: str = "pred_cyc",
        sar_sub: str = "A",
        resize: Tuple[int, int] = (256, 256),
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.opt_dir = self.root_dir / split / optical_sub
        self.sar_dir = self.root_dir / split / sar_sub
        if not self.opt_dir.is_dir() or not self.sar_dir.is_dir():
            raise FileNotFoundError(f"Expect folders: {self.opt_dir} and {self.sar_dir}")

        opt_files = sorted(self.opt_dir.glob("*.png"))
        pairs = []
        for opt_path in opt_files:
            sar_name = opt_path.name.replace("_s2_", "_s1_")
            sar_path = self.sar_dir / sar_name
            if not sar_path.exists():
                raise FileNotFoundError(f"Pair for {opt_path.name} not found in {self.sar_dir}")
            pairs.append((sar_path, opt_path))
        self.pairs = pairs

        self.sar_transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.opt_transform = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        sar_path, opt_path = self.pairs[idx]
        sar_img = Image.open(sar_path).convert("L")
        opt_img = Image.open(opt_path).convert("RGB")

        sar_t = self.sar_transform(sar_img)   # [1, H, W]
        opt_t = self.opt_transform(opt_img)   # [3, H, W]
        x = torch.cat([sar_t, opt_t], dim=0)  # [4, H, W]
        # x = opt_t

        cls_name = parse_class_from_filename(opt_path.name)
        y_idx = NAME_TO_IDX[cls_name]
        y_onehot = make_one_hot(y_idx, len(CLASS_NAMES))

        return {
            "image": x,
            "label_index": torch.tensor(y_idx, dtype=torch.long),
            "label_onehot": y_onehot,
            "path_opt": str(opt_path),
            "path_sar": str(sar_path),
        }


def build_loader(root_dir: str, split: str, batch_size=32, num_workers=4, resize=(256, 256), shuffle=None):
    ds = PairedRSCDataset(root_dir=root_dir, split=split, resize=resize)
    if shuffle is None:
        shuffle = (split.lower() == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


# # ====== 模型：ResNet-50 无预训练，4通道输入 ======
# def build_model(in_ch=4, num_classes=6) -> nn.Module:
#     model = models.resnet50(weights=None)  # 不加载预训练
#     # 替换 conv1 以支持 4 通道输入
#     old_conv: nn.Conv2d = model.conv1
#     new_conv = nn.Conv2d(in_ch, old_conv.out_channels,
#                          kernel_size=old_conv.kernel_size,
#                          stride=old_conv.stride,
#                          padding=old_conv.padding,
#                          bias=False)
#     # Kaiming 初始化
#     nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
#     model.conv1 = new_conv
#
#     # 替换分类头
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)
#     nn.init.normal_(model.fc.weight, std=0.01)
#     nn.init.zeros_(model.fc.bias)
#
#     return model
#
#
# # ====== 模型：ResNet-18 无预训练，4通道输入 ======
# def build_model_resnet18(in_ch=4, num_classes=6) -> nn.Module:
#     model = models.resnet18(weights=None)  # 不加载预训练
#
#     # 替换 conv1 以支持 4 通道输入
#     old_conv: nn.Conv2d = model.conv1
#     new_conv = nn.Conv2d(
#         in_ch,
#         old_conv.out_channels,
#         kernel_size=old_conv.kernel_size,
#         stride=old_conv.stride,
#         padding=old_conv.padding,
#         bias=False
#     )
#     # Kaiming 初始化
#     nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
#     model.conv1 = new_conv
#
#     # 替换分类头
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)
#     nn.init.normal_(model.fc.weight, std=0.01)
#     nn.init.zeros_(model.fc.bias)
#
#     return model
#
#
# # ====== 模型：ResNet-34 无预训练，4通道输入 ======
# def build_model_resnet34(in_ch=4, num_classes=6) -> nn.Module:
#     model = models.resnet34(weights=None)  # 不加载预训练
#
#     # 替换 conv1 以支持 4 通道输入
#     old_conv: nn.Conv2d = model.conv1
#     new_conv = nn.Conv2d(
#         in_ch,
#         old_conv.out_channels,
#         kernel_size=old_conv.kernel_size,
#         stride=old_conv.stride,
#         padding=old_conv.padding,
#         bias=False
#     )
#     # Kaiming 初始化
#     nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
#     model.conv1 = new_conv
#
#     # 替换分类头
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)
#     nn.init.normal_(model.fc.weight, std=0.01)
#     nn.init.zeros_(model.fc.bias)
#
#     return model


# ====== 训练 / 验证 ======

# ====== 统一的模型构建函数：支持 resnet18 / resnet34 / resnet50，4通道输入 ======
def build_model(arch: str = "resnet18", in_ch: int = 4, num_classes: int = 6) -> nn.Module:
    """
    arch: "resnet18", "resnet34", "resnet50"
    """
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet34":
        model = models.resnet34(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    # -------- 替换 conv1，支持 4 通道输入 --------
    old_conv: nn.Conv2d = model.conv1
    new_conv = nn.Conv2d(
        in_ch,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
    model.conv1 = new_conv

    # -------- 替换分类头 --------
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    nn.init.normal_(model.fc.weight, std=0.01)
    nn.init.zeros_(model.fc.bias)

    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y_idx = batch["label_index"].to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y_idx)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y_idx).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y_idx = batch["label_index"].to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y_idx)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y_idx).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/NAS_data/yjy/SEN1-2-BIT_matched",
                        help="/NAS_data/yjy/SEN1-2-BIT_matched")
    parser.add_argument("--gpu", type=int, default=3, choices=list(range(8)), help="GPU index 0-7")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # 只使用指定 GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # 这里用 cuda:0，因为 CUDA_VISIBLE_DEVICES 已经把外面的 ID 映射进来了
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})")

    # ===== DataLoader 只建一次，两个模型共用 =====
    train_loader = build_loader(args.root, "Train", batch_size=args.batch_size,
                                num_workers=args.num_workers, resize=(256, 256))
    test_loader = build_loader(args.root, "Test", batch_size=args.batch_size,
                               num_workers=args.num_workers, resize=(256, 256), shuffle=False)

    # 要跑的模型列表：这里你想测哪个就写哪个
    model_names = ["resnet18", "resnet34"]
    save_dir = "classification_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for arch in model_names:
        print("\n" + "=" * 80)
        print(f"🔹 Start training model: {arch}")
        print("=" * 80)

        # ===== 构建模型和优化器 =====
        model = build_model(arch=arch, in_ch=4, num_classes=len(CLASS_NAMES)).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_acc = 0.0
        best_path = f"{save_dir}/best_{arch}_sar_pred_cyc.pth"

        # 该模型对应的 metrics 子目录
        metrics_dir = f"metrics/{arch}"
        os.makedirs(metrics_dir, exist_ok=True)

        # ===== 训练多轮 =====
        for ep in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
            te_loss, te_acc = evaluate(model, test_loader, device)

            print(f"[{arch}] Epoch {ep:02d} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"test loss {te_loss:.4f} acc {te_acc:.4f}")

            # 保存 best checkpoint（每个 arch 各自一份）
            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(model.state_dict(), best_path)
                print(f"✅ [{arch}] Saved best model at epoch {ep} with acc={best_acc:.4f} → {best_path}")

            # 计算并保存该 epoch 的各类指标（放到 metrics/arch/ 下面）
            logits, targets = gather_logits_targets(model, test_loader, device)
            compute_and_save_metrics(
                class_names=CLASS_NAMES,
                save_dir=metrics_dir,
                prefix=f"{arch}_epoch{ep:02d}_test",  # 文件带模型名前缀，防止互相覆盖
                logits=logits,
                targets=targets
            )

        print(f"✅ Finished training {arch}, best test acc: {best_acc:.4f}")



if __name__ == "__main__":
    main()