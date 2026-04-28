import cv2
import os
import numpy as np
def adaptive_equalize_canny_image(input_path, input_path_2,output_dir="/data/yjy_data/DDBM_GT_Unet/canny_optimization_result_SEN12_scene/test_result_120/out_clahe"):
    """
    对Canny算子图像进行自适应直方图均衡化（CLAHE）并保存结果。

    Args:
        input_path (str): Canny图像路径。
        output_dir (str): 输出目录。

    Returns:
        output_path (str): 均衡化后图像的保存路径。
    """
    # 1️⃣ 读取图像（灰度模式）
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"❌ 无法读取图像：{input_path}")

    img2 = cv2.imread(input_path_2, cv2.IMREAD_GRAYSCALE)
    if img2 is None:
        raise FileNotFoundError(f"❌ 无法读取图像：{input_path}")
    # 2️⃣ 创建 CLAHE 对象（clipLimit 控制对比度限制，tileGridSize 控制分块大小）
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 3️⃣ 应用自适应直方图均衡
    # equalized = clahe.apply(img)
    SAR_pred = 0.5*img + 0.5*img2
    SAR_pred = np.clip(SAR_pred, 0, 255).astype(np.uint8)
    # 4️⃣ 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 5️⃣ 保存结果
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"fusion_{filename}")
    cv2.imwrite(output_path, SAR_pred)

    print(f"✅ 自适应直方图均衡完成，保存路径：{output_path}")
    return output_path


# 🔧 示例调用
adaptive_equalize_canny_image("/data/yjy_data/DDBM_GT_Unet/canny_optimization_result_SEN12_scene/test_result_120/pred_edge/ROIs1158_spring_s1_47_p106_pred_edge.png","/data/yjy_data/DDBM_GT_Unet/canny_optimization_result_SEN12_scene/test_result_120/ROIs1158_spring_s1_47_p106_sar_edge.png")
# adaptive_equalize_canny_image("/data/yjy_data/DDBM_GT_Unet/canny_optimization_result_SAR2OPT_v2/test_result_120_all/out_CLAHE/fusion_11_1440_240_pred_edge.png")