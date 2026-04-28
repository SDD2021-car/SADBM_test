import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
def canny_edge_detection(image, low_threshold=20, high_threshold=70):

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

# 读取输入图像
input_image = cv2.imread('/data/yjy_data/SAR_denoise/Train_data_Results/SAR_denoised/denoise_2_1920_2640.jpg')  # 替换为你自己的图片路径
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) / 255.0  # 归一化

# 应用Canny边缘检测
output_edges = canny_edge_detection(input_image_rgb)
output_image = (output_edges.squeeze().cpu().numpy() * 255).astype(np.uint8)
output_image = np.transpose(output_image, (1, 2, 0))
# 检查文件夹是否存在
output_path = '/data/yjy_data/canny_edges_output.jpg'
output_folder = os.path.dirname(output_path)

if not os.path.exists(output_folder):
    print(f"Output folder does not exist: {output_folder}")
else:
    print(f"Output folder exists: {output_folder}")

# 检查保存图像
if cv2.imwrite(output_path, output_image):
    print(f"Image successfully saved to {output_path}")
else:
    print(f"Failed to save image to {output_path}")
