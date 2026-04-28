import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def add_title(img, title_text, bar_height=40):
    """
    在图像顶部添加一块文字栏，并写上对应的方法名称
    """
    h, w = img.shape[:2]

    # 顶部白色栏（如果想黑色就改成全0）
    bar = np.ones((bar_height, w, 3), dtype=np.uint8) * 255

    # 在 bar 上写字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    # 文字基线位置（左对齐，稍微往下居中一点）
    text_org = (10, int(bar_height * 0.75))

    cv2.putText(
        bar,
        title_text,
        text_org,
        font,
        font_scale,
        (0, 0, 0),   # 黑色字体
        thickness,
        cv2.LINE_AA
    )

    # 上下拼接：文字栏 + 原图
    img_with_title = np.vstack([bar, img])
    return img_with_title


def resize_and_concatenate(paths, path_AB, titles):
    """
    读取多张图片 -> 统一尺寸 -> 每张加标题 -> 横向拼接 -> 保存
    """
    images = [cv2.imread(path, 1) for path in paths]

    # 简单防御一下：如果有图片读取失败
    if any(img is None for img in images):
        print(f"[Warning] Some images failed to load, skip: {paths}")
        return

    # 获取第一张图片的高度和宽度，用作目标尺寸
    h, w = images[0].shape[:2]

    # 调整所有图片的大小，使它们具有相同的尺寸
    resized_images = [cv2.resize(img, (w, h)) for img in images]

    # 给每张图片添加标题
    images_with_titles = [
        add_title(img, title)
        for img, title in zip(resized_images, titles)
    ]

    # 水平拼接所有图片
    im_AB = np.concatenate(images_with_titles, axis=1)

    # 保存合并后的图片
    cv2.imwrite(path_AB, im_AB)


parser = argparse.ArgumentParser('create image pairs')
# GT
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image folder A', type=str,
                    default="/data/yjy_data/Pix2Pix_S2O/result/experiment_pix_GF3_scene/test_200/fakeB")
# pix2pix
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image folder B', type=str,
                    default="/data/yjy_data/pix2pix_and_cyclegan_0808/result_GF3/experiment_cyc_GF3_0320/test_200/images")
# CycleGAN
parser.add_argument('--fold_C', dest='fold_C', help='input directory for image folder C', type=str,
                    default="/data/yjy_data/pix2pix_and_cyclegan_0808/result_GF3/experiment_scyc_GF3_0320/test_200/images")
# S-CycleGAN
parser.add_argument('--fold_D', dest='fold_D', help='input directory for image folder D', type=str,
                    default="/NAS_data/cyh/S2O_baselines/CUT_new/results/GF3_sar2opt_2025.03.31/test_200/images/fake_B")

# ASGIT
parser.add_argument('--fold_E', dest='fold_E', help='input directory for image folder E', type=str,
                    default="/NAS_data/cyh/S2O_baselines/ASGIT/results/GF3_sar2opt_2025.04.17/test_200/images")

# CUT
parser.add_argument('--fold_F', dest='fold_F', help='input directory for image folder F', type=str,
                    default="/data/yjy_data/Parallel_GAN_correct/results/Para_G_GF3/images")

# Parallel
parser.add_argument('--fold_G', dest='fold_G', help='input directory for image folder G', type=str,
                    default="/NAS_data/cyh/DOGAN_all_determined_experiment/results/DOGAN_GF3_5.26_remake1/images")

# Palette
parser.add_argument('--fold_H', dest='fold_H', help='input directory for image folder H', type=str,
                    default="/data/cyh/Datasets/GF3_sar2opt/testB")

# DDBM_GT_se / COGAN（你原来的 I）
parser.add_argument('--fold_I', dest='fold_I', help='input directory for image folder I', type=str,
                    default="/data/cyh/Datasets/GF3_sar2opt/testA")

parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str,
                    default="/NAS_data/yjy/GF3_combine_results")

parser.add_argument('--num_imgs', dest='num_imgs', help='number of image groups', type=int, default=1000000)
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing',
                    help='If used, chooses single CPU execution instead of parallel execution', action='store_true',
                    default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

# 获取文件夹内的文件并进行排序
img_list_A = sorted(os.listdir(args.fold_A))
img_list_B = sorted(os.listdir(args.fold_B))
img_list_C = sorted(os.listdir(args.fold_C))
img_list_D = sorted(os.listdir(args.fold_D))
img_list_E = sorted(os.listdir(args.fold_E))
img_list_F = sorted(os.listdir(args.fold_F))
img_list_G = sorted(os.listdir(args.fold_G))
img_list_H = sorted(os.listdir(args.fold_H))
img_list_I = sorted(os.listdir(args.fold_I))

num_imgs = min(args.num_imgs, len(img_list_A))
img_fold_AB = args.fold_AB
if not os.path.isdir(img_fold_AB):
    os.makedirs(img_fold_AB)

# 对应每一列图像的方法名称（顺序要和 all_paths 一一对应）
titles = [
    "Pix2Pix",          # A
    "CycleGAN",     # B
    "S-CycleGAN",    # C
    "CUT",          # E
    "ASGIT",       # F
    "Parallel",    # G
    "DOGAN",     # H
    "GT",        # I（你可以改成你想展示的名字）
    "SAR"
]

if not args.no_multiprocessing:
    pool = Pool()

for n in range(num_imgs):
    # 从每个文件夹中取一张图片
    name_A = img_list_A[n]
    name_B = img_list_B[n]
    name_C = img_list_C[n]
    name_D = img_list_D[n]
    name_E = img_list_E[n]
    name_F = img_list_F[n]
    name_G = img_list_G[n]
    name_H = img_list_H[n]
    name_I = img_list_I[n]

    # 获取每张图片的路径
    path_A = os.path.join(args.fold_A, name_A)
    path_B = os.path.join(args.fold_B, name_B)
    path_C = os.path.join(args.fold_C, name_C)
    path_D = os.path.join(args.fold_D, name_D)
    path_E = os.path.join(args.fold_E, name_E)
    path_F = os.path.join(args.fold_F, name_F)
    path_G = os.path.join(args.fold_G, name_G)
    path_H = os.path.join(args.fold_H, name_H)
    path_I = os.path.join(args.fold_I, name_I)

    # 将 9 张图片的路径放入一个列表（顺序要和 titles 对齐）
    all_paths = [path_A, path_B, path_C, path_D, path_E, path_F, path_G, path_H, path_I]

    # 生成合并后的图片路径
    name_AB = name_A  # 图片命名格式可以根据需要调整
    path_AB = os.path.join(img_fold_AB, name_AB)

    # 检查文件是否存在
    if all(os.path.isfile(path) for path in all_paths):
        if not args.no_multiprocessing:
            pool.apply_async(resize_and_concatenate, args=(all_paths, path_AB, titles))
        else:
            resize_and_concatenate(all_paths, path_AB, titles)

if not args.no_multiprocessing:
    pool.close()
    pool.join()
