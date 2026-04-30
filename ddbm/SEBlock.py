import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = torch.nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, (1, 1))
        squeeze = squeeze.view(batch_size, channels)
        excitation = F.relu(self.fc1(squeeze))
        excitation = self.sigmoid(self.fc2(excitation))
        excitation = excitation.view(batch_size, channels, 1, 1)
        return x * excitation.expand_as(x)


class FeatureFusionNet(nn.Module):
    def __init__(self, use_fp16: bool = False, device: Optional[torch.device] = None):
        super(FeatureFusionNet, self).__init__()

        # 卷积层
        self.conv_1 = torch.nn.Conv2d(128, 128, kernel_size=1)
        self.conv_2 = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.conv_3 = torch.nn.Conv2d(512, 128, kernel_size=1)
        self.conv_4 = torch.nn.Conv2d(512, 128, kernel_size=1)

        # SE模块
        self.se_block = SEBlock(in_channels=128 * 4)

        # 最终卷积层
        self.final_conv = torch.nn.Conv2d(512, 512, kernel_size=1)
        # 统一处理精度 / 设备
        if use_fp16:
            self.half()
        if device is not None:
            self.to(device)
    def forward(self, x1, x2, x3, x4):
        # 尽量对齐到模块参数的 device / dtype；若当前模块无参数，则回退到输入张量。
        first_param = next(self.parameters(), None)
        if first_param is not None:
            device = first_param.device
            dtype = first_param.dtype
        else:
            device = x1.device
            dtype = x1.dtype

        x1 = x1.to(device=device, dtype=dtype)
        x2 = x2.to(device=device, dtype=dtype)
        x3 = x3.to(device=device, dtype=dtype)
        x4 = x4.to(device=device, dtype=dtype)
        # 调整特征图尺寸一致
        x1_up = F.interpolate(x1, size=(8, 8), mode='bilinear', align_corners=False)
        x2_up = F.interpolate(x2, size=(8, 8), mode='bilinear', align_corners=False)
        x3_up = F.interpolate(x3, size=(8, 8), mode='bilinear', align_corners=False)
        x4_up = F.interpolate(x4, size=(8, 8), mode='bilinear', align_corners=False)

        # 使用1x1卷积调整通道数
        x1_up = self.conv_1(x1_up)
        x2_up = self.conv_2(x2_up)
        x3_up = self.conv_3(x3_up)
        x4_up = self.conv_4(x4_up)

        # 拼接四个特征图
        fused = torch.cat([x1_up, x2_up, x3_up, x4_up], dim=1)

        # SE模块加权融合
        output = self.se_block(fused)

        # 最终卷积调整通道数
        output = self.final_conv(output)

        return output

    def save_checkpoint(self, path):
        """保存模型的checkpoint"""
        torch.save(self.state_dict(), path)

    def load_checkpoint(
        self,
        path: str,
        device: Optional[torch.device] = None,
        use_fp16: Optional[bool] = None,
    ):
        """
        加载 checkpoint：
        - 先加载到 CPU，再根据参数选择是否搬到指定 device 和 half
        """
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)

        # 按需设置精度 / 设备
        if use_fp16 is True:
            self.half()
        elif use_fp16 is False:
            self.float()

        if device is not None:
            self.to(device)

        self.eval()


# 使用示例
# device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# # 实例化网络
# model = FeatureFusionNet(device)
#
# # 输入的四个特征图，假设它们的尺寸分别是(10, 128, 256, 256), (10, 256, 64, 64), (10, 512, 16, 16), (10, 512, 8, 8)
# x1 = torch.randn(10, 128, 256, 256).to(device)
# x2 = torch.randn(10, 256, 64, 64).to(device)
# x3 = torch.randn(10, 512, 16, 16).to(device)
# x4 = torch.randn(10, 512, 8, 8).to(device)
#
# # 调用模型
# output = model(x1, x2, x3, x4)
#
# # 输出结果的形状
# print(f'Output shape: {output.shape}')
#
# # 保存模型 checkpoint
# model.save_checkpoint('model_checkpoint.pth')
#
# # 加载模型 checkpoint
# model.load_checkpoint('model_checkpoint.pth')
