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