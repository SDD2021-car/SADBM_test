import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# 1️⃣ Dataset
# ============================================================
class SAROptDataset(Dataset):
    def __init__(self, sar_dir, opt_dir, transform=None):
        self.sar_files = sorted(glob.glob(os.path.join(sar_dir, "*.png")))
        self.opt_files = sorted(glob.glob(os.path.join(opt_dir, "*.png")))
        assert len(self.sar_files) == len(self.opt_files), "数据数量不匹配"
        self.transform = transform

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar = Image.open(self.sar_files[idx]).convert("RGB")
        opt = Image.open(self.opt_files[idx]).convert("RGB")
        if self.transform:
            sar = self.transform(sar)
            opt = self.transform(opt)
        return sar, opt


# ============================================================
# 2️⃣ GPU版 Sobel 边缘提取函数
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
    edges_max = edges.flatten(1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    edges = edges / (edges.max() + 1e-8)
    return edges.mean(dim=1, keepdim=True)  # [B,1,H,W]


# ============================================================
# 3️⃣ U-Net 模型定义
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
# 4️⃣ 训练配置
# ============================================================
if __name__ == "__main__":  # ✅ 只在直接运行时执行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = SAROptDataset(
        sar_dir="/data/yjy_data/dataset/sen_data_new/trainA/",
        opt_dir="/data/yjy_data/dataset/sen_data_new/trainB/",
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 200
    save_dir = "/data/yjy_data/DDBM_GT_Unet/canny_optimization_result"
    os.makedirs(save_dir, exist_ok=True)

# ============================================================
# 5️⃣ 训练循环（带tqdm进度条）
# ============================================================

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for sar, opt in progress_bar:
            sar, opt = sar.to(device, non_blocking=True), opt.to(device, non_blocking=True)
            sar_edge = gpu_edge_approx(sar)
            opt_edge = gpu_edge_approx(opt)

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
                sar, opt = next(iter(train_loader))
                sar, opt = sar.to(device), opt.to(device)
                sar_edge = gpu_edge_approx(sar)
                opt_edge = gpu_edge_approx(opt)
                pred = model(sar_edge)

            sar_edge, opt_edge, pred = sar_edge.cpu(), opt_edge.cpu(), pred.cpu()
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            for i in range(3):
                axes[i, 0].imshow(sar_edge[i][0], cmap='gray')
                axes[i, 0].set_title('Input SAR-Edge')
                axes[i, 1].imshow(pred[i][0].detach(), cmap='gray')
                axes[i, 1].set_title('Predicted Edge')
                axes[i, 2].imshow(opt_edge[i][0], cmap='gray')
                axes[i, 2].set_title('GT Optical-Edge')
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
