import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(SimpleResNetFeatureExtractor, self).__init__()

        # 第一层卷积：输出通道数64，卷积核3x3，步幅1，padding为1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 第二层卷积：输出通道数64，卷积核3x3，步幅1，padding为1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 第一层残差块（Skip Connection）
        self.res1 = self._make_residual_block(64, 64)

        # 第三层卷积：输出通道数128，卷积核3x3，步幅1，padding为1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 第四层卷积：输出通道数128，卷积核3x3，步幅1，padding为1
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        # 第二层残差块（Skip Connection）
        self.res2 = self._make_residual_block(128, 128)

        # 池化层：2x2最大池化，步幅2，输出尺寸减半
        self.pool = nn.MaxPool2d(2, 2)

        # 第五层卷积：输出通道数256，卷积核3x3，步幅1，padding为1
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 第六层卷积：输出通道数256，卷积核3x3，步幅1，padding为1
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # 第三层残差块（Skip Connection）
        self.res3 = self._make_residual_block(256, 256)

        # 全连接层，用于将提取的特征转换为一个固定大小的特征向量
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)  # 全连接层1
        self.fc2 = nn.Linear(1024, 256)  # 输出一个256维的特征向量

        # 上采样层，用于将特征图恢复到与输入图像相同的尺寸
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

    def _make_residual_block(self, in_channels, out_channels):
        """ 创建一个残差块 """
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # 第一层卷积 + ReLU 激活
        x1 = F.relu(self.conv1(x))
        # 第二层卷积 + ReLU 激活
        x2 = F.relu(self.conv2(x1))

        # 第一层残差块 + Skip Connection
        x_res1 = self.res1(x2) + x2

        # 第三层卷积 + ReLU 激活
        x3 = F.relu(self.conv3(x_res1))
        # 第四层卷积 + ReLU 激活
        x4 = F.relu(self.conv4(x3))

        # 第二层残差块 + Skip Connection
        x_res2 = self.res2(x4) + x4

        # 池化
        x_pooled = self.pool(x_res2)

        # 第五层卷积 + ReLU 激活
        x5 = F.relu(self.conv5(x_pooled))
        # 第六层卷积 + ReLU 激活
        x6 = F.relu(self.conv6(x5))

        # 第三层残差块 + Skip Connection
        x_res3 = self.res3(x6) + x6

        # 展平特征图，准备进入全连接层
        x_flat = x_res3.view(-1, 256 * 16 * 16)  # 将特征图展平为一维向量

        # 全连接层1
        x_fc1 = F.relu(self.fc1(x_flat))
        # 全连接层2
        x_fc2 = self.fc2(x_fc1)

        # 上采样，将特征图恢复到与输入图像相同的尺寸
        upsampled_feature_map = self.upsample(x_res3)

        # 拼接原始图像和上采样后的特征图
        fused_output = torch.cat((x, upsampled_feature_map), dim=1)  # 沿通道维度拼接

        return x_fc2, fused_output  # 返回特征向量和融合后的图像


# 创建模型
model = SimpleResNetFeatureExtractor()

# 打印模型结构
print(model)

# 假设输入图像大小是256x256, 3通道（RGB图像）
input_image = torch.randn(1, 3, 256, 256)  # 批次大小1
output_features, fused_output = model(input_image)

print("输出特征向量的形状：", output_features.shape)
print("融合后的输出形状：", fused_output.shape)  # 融合后的图像形状

