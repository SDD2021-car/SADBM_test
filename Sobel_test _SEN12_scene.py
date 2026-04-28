import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms.functional as TF
# ============================================================
# 1️⃣ Dataset（测试集版）
# ============================================================
class SAROptDataset(Dataset):
    def __init__(self, sar_dir, opt_dir, transform=None):
        self.sar_files = sorted(glob.glob(os.path.join(sar_dir, "*.png")))
        self.opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.png")))
        assert len(self.sar_files) == len(self.opt_files), "测试集数量不匹配"
        self.transform = transform

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar = Image.open(self.sar_files[idx]).convert("RGB")
        opt = Image.open(self.opt_files[idx]).convert("RGB")
        if self.transform:
            sar = self.transform(sar)
            opt = self.transform(opt)
        return sar, opt, os.path.basename(self.sar_files[idx])

# ============================================================
# 2️⃣ GPU版 Sobel 边缘提取（保持与训练一致）
# ============================================================
def gpu_edge_approx(images):
    """
    输入: [B,3,H,W] Tensor
    输出: [B,1,H,W] Tensor (归一化边缘图)
    """
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=images.device)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32, device=images.device)
    sobel_x = sobel_x.expand(images.size(1), 1, 3, 3)
    sobel_y = sobel_y.expand(images.size(1), 1, 3, 3)
    grad_x = nn.functional.conv2d(images, sobel_x, padding=1, groups=images.size(1))
    grad_y = nn.functional.conv2d(images, sobel_y, padding=1, groups=images.size(1))
    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    edges = edges / (edges.max() + 1e-8)
    return edges.mean(dim=1, keepdim=True)

# ============================================================
# 3️⃣ U-Net 模型定义（完全一致）
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
# 4️⃣ 推理函数
# ============================================================
def test_model(model_path, sar_dir, opt_dir, save_dir, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    test_dataset = SAROptDataset(sar_dir, opt_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 模型加载
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 推理
    # with torch.no_grad():
    #     for sar, opt, names in tqdm(test_loader, desc="Testing (GPU)", leave=True):
    #         sar, opt = sar.to(device, non_blocking=True), opt.to(device, non_blocking=True)
    #         sar_edge = gpu_edge_approx(sar)
    #         opt_edge = gpu_edge_approx(opt)
    #         pred = model(sar_edge)
    #
    #         sar_edge, opt_edge, pred = sar_edge.cpu(), opt_edge.cpu(), pred.cpu()
    #
    #         for i in range(sar.size(0)):
    #             fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    #             axes[0].imshow(sar_edge[i][0], cmap='gray')
    #             axes[0].set_title('Input SAR Edge')
    #             axes[1].imshow(pred[i][0], cmap='gray')
    #             axes[1].set_title('Predicted Edge')
    #             axes[2].imshow(opt_edge[i][0], cmap='gray')
    #             axes[2].set_title('GT Optical Edge')
    #             for ax in axes:
    #                 ax.axis('off')
    #             plt.tight_layout()
    #             filename = names[i].replace('.png', '_compare.png')
    #             plt.savefig(os.path.join(save_dir, filename))
    #             plt.close()
    # 推理
    with torch.no_grad():
        for sar, opt, names in tqdm(test_loader, desc="Testing (GPU)", leave=True):
            sar, opt = sar.to(device, non_blocking=True), opt.to(device, non_blocking=True)
            sar_edge = gpu_edge_approx(sar)
            opt_edge = gpu_edge_approx(opt)
            pred = model(sar_edge)

            # 移到CPU
            sar, opt, sar_edge, opt_edge, pred = sar.cpu(), opt.cpu(), sar_edge.cpu(), opt_edge.cpu(), pred.cpu()

            for i in range(sar.size(0)):
                base_name = os.path.splitext(names[i])[0]

                # 🔹保存原始SAR图像
                TF.to_pil_image(sar[i]).save(os.path.join(save_dir, f"{base_name}_sar.png"))

                # 🔹保存原始Optical图像
                TF.to_pil_image(opt[i]).save(os.path.join(save_dir, f"{base_name}_opt.png"))
                # 保存SAR边缘
                TF.to_pil_image(sar_edge[i]).save(os.path.join(save_dir, f"{base_name}_sar_edge.png"))
                # 🔹保存预测结果
                TF.to_pil_image(pred[i]).save(os.path.join(save_dir, f"{base_name}_pred_edge.png"))

                # 🔹保存GT边缘图
                TF.to_pil_image(opt_edge[i]).save(os.path.join(save_dir, f"{base_name}_gt_edge.png"))
    print(f"✅ Testing finished! Results saved to: {save_dir}")

# ============================================================
# 5️⃣ 主函数入口
# ============================================================
if __name__ == "__main__":
    test_model(
        model_path="/data/yjy_data/DDBM_GT_Unet/canny_optimization_result_SEN12_scene/unet_epoch_120.pth",
        sar_dir="/data/hjf/Dataset/SEN12_Scene/testA/",
        opt_dir="/data/hjf/Dataset/SEN12_Scene/testB/",
        save_dir="/data/yjy_data/DDBM_GT_Unet/canny_optimization_result_SEN12_scene/test_result_120_v2/",
        batch_size=4
    )
