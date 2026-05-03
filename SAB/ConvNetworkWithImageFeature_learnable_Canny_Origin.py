import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import Canny_train


def canny_edge_detection(image, low_threshold=20, high_threshold=150):

    # 确保输入是numpy数组
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # 处理批次
    if image.ndim == 4:
        # 假设批次维度在第一个维度
        return np.stack([canny_edge_detection(img, low_threshold, high_threshold) for img in image])

    # 转换为BGR格式（如果是RGB）
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # 确保图像是8位无符号整数类型
    image = (image * 255).astype(np.uint8)

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 应用Canny边缘检测
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # 将结果转换回torch tensor
    return torch.from_numpy(edges).unsqueeze(0).repeat(3, 1, 1).float() / 255.0


def apply_canny_to_batch(batch_image):
    # 假设 batch_image 的形状是 (batch_size, channels, height, width)
    batch_size = batch_image.shape[0]
    canny_outputs = []

    for i in range(batch_size):
        canny_output = canny_edge_detection(batch_image[i])
        canny_outputs.append(canny_output)

    # 将所有的 Canny 输出堆叠成一个批次
    return torch.stack(canny_outputs, dim=0)



class ConvNetworkWithImageFeature(nn.Module):
    def __init__(self, in_channels, out_channels, fusion_canny, **kwargs):
        super(ConvNetworkWithImageFeature, self).__init__()

        self.ratio = 0.00001
        # 第一层卷积: in_channels -> 64 通道
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.silu1 = nn.SiLU()

        # 第二层卷积: 64 -> 128 通道
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.silu2 = nn.SiLU()

        # 第三层卷积: 128 -> 64 通道
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.silu3 = nn.SiLU()

        # 第四层卷积: 64 -> out_channels 通道
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        # 1x1 卷积，用于调整输入 x 的通道数，以便进行跳跃连接
        # self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_feature1 = nn.Conv2d(3, 64, kernel_size=1)
        self.conv_feature2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv_feature3 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_feature4 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.stage0_train = False  # False是分阶段训练 True是联合训练
        self.fusion_canny = fusion_canny  # False是不融合提质前后的canny True是融合

        # 可学习的融合参数：raw_alpha_fusion ∈ (-∞, +∞)
        # 通过 sigmoid 映射到 (0,1) 作为 alpha0_fusion
        # NOTE: 初始化为 0 ⇒ sigmoid(0)=0.5，你也可以用 logit(0.9) 等作为初值
        self.raw_alpha_fusion = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # canny提纯网络实例化
        self.canny_refine_network = Canny_train.UNet(in_channels=1, out_channels=1)

        if self.stage0_train:
            pass
        else:
            path_refine_network = '/NAS_data/yjy/DDBM_GT_Unet/DDBM_S2O_Canny_Original_results/unet_epoch_120.pth'
            # 加载预训练权重
            checkpoint = torch.load(path_refine_network, map_location='cpu')  # 或者 'cuda:7' 根据你的设备
            # 支持两种情况：保存的是整个模型或 state_dict
            if 'state_dict' in checkpoint:
                self.canny_refine_network.load_state_dict(checkpoint['state_dict'])
            else:
                self.canny_refine_network.load_state_dict(checkpoint)

            # 冻结参数（不参与反向传播）
            for param in self.canny_refine_network.parameters():
                param.requires_grad = False

            # 切换到 eval 模式（关闭 Dropout / BatchNorm 统计更新）
            self.canny_refine_network.eval()

    def convert_to_fp16(self):
        # 将模型转换为半精度（FP16）
        for param in self.parameters():
            param.data = param.data.half()
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.float()  # BatchNorm不应该使用FP16

    def forward(self, x, alpha):
        # 对输入图像进行Canny边缘检测
        device = x.device
        SAR_canny_without_refine = apply_canny_to_batch(x).to(device)
        SAR_canny_with_refine = self.canny_refine_network(
            SAR_canny_without_refine.mean(dim=1, keepdim=True)
        )

        if self.fusion_canny:
            # 将可学习参数 raw_alpha_fusion 限制到 (0,1)
            alpha0 = torch.sigmoid(self.raw_alpha_fusion)
            alpha1 = 1.0 - alpha0

            # 这里 alpha0/alpha1 是标量 tensor，会自动 broadcast 到整个 batch
            SAR_canny_with_refine_fusion = (
                    alpha0 * SAR_canny_without_refine
                    + alpha1 * SAR_canny_with_refine
            )
        else:
            SAR_canny_with_refine_fusion = SAR_canny_with_refine

        SAR_canny_with_refine_fusion = SAR_canny_with_refine_fusion.to(device)
        canny_resized = SAR_canny_with_refine_fusion * alpha[0]
        # 第一层卷积

        x1 = self.conv1(x + canny_resized)
        x1 = self.silu1(x1)

        # 第二层卷积
        canny_resized = self.conv_feature1(canny_resized)
        x2 = self.conv2(x1 + canny_resized * alpha[1])
        x2 = self.silu2(x2)

        # 第三层卷积
        canny_resized = self.conv_feature2(canny_resized)
        x3 = self.conv3(x2 + canny_resized * alpha[2])
        x3 = self.silu3(x3)

        # 第四层卷积
        canny_resized = self.conv_feature3(canny_resized)
        x4 = self.conv4(x3 + canny_resized * alpha[3])*self.ratio
        x4 = x4 + x
        output = x4

        if self.stage0_train:
            return output, SAR_canny_with_refine_fusion
        else:
            return output
