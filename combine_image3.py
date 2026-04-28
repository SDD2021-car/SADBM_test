import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def resize_and_concatenate(paths, path_AB):
    images = [cv2.imread(path, 1) for path in paths]

    # 获取第一张图片的高度和宽度，用作目标尺寸
    h, w = images[0].shape[:2]

    # 调整所有图片的大小，使它们具有相同的尺寸
    resized_images = [cv2.resize(img, (w, h)) for img in images]

    # 水平拼接所有图片
    im_AB = np.concatenate(resized_images, axis=1)

    # 保存合并后的图片
    cv2.imwrite(path_AB, im_AB)


parser = argparse.ArgumentParser('create image pairs')
# GT
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image folder A', type=str,
                    default="/data/yjy_data/pix2pix_and_cyclegan_0808/result_SAR2OPT/s_cyclegan_bs2/experiment_scyc_SAR2OPT_bs2/test_200/realB")
# DDBM
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image folder B', type=str,
                    default="/data/yjy_data/DDBM/result_SAR2OPT_512")
# DDBM_GT
parser.add_argument('--fold_C', dest='fold_C', help='input directory for image folder C', type=str,
                    default="/data/yjy_data/DDBM/result_SAR2OPT_GT_0126")
# DDBM_GT_se
parser.add_argument('--fold_D', dest='fold_D', help='input directory for image folder D', type=str,
                    default="/data/yjy_data/DDBM/result_SAR2OPT_GT_se1_0208")

parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str,
                    default="/data/yjy_data/DDBM_GT_Unet/combine_3_test")

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
# img_list_E = sorted(os.listdir(args.fold_E))
# img_list_F = sorted(os.listdir(args.fold_F))
# img_list_G = sorted(os.listdir(args.fold_G))
# img_list_H = sorted(os.listdir(args.fold_H))
# img_list_I = sorted(os.listdir(args.fold_I))

num_imgs = min(args.num_imgs, len(img_list_A))
img_fold_AB = args.fold_AB
if not os.path.isdir(img_fold_AB):
    os.makedirs(img_fold_AB)

if not args.no_multiprocessing:
    pool = Pool()

for n in range(num_imgs):
    # 从每个文件夹中取一张图片
    name_A = img_list_A[n]
    name_B = img_list_B[n]
    name_C = img_list_C[n]
    name_D = img_list_D[n]
    # name_E = img_list_E[n]
    # name_F = img_list_F[n]
    # name_G = img_list_G[n]
    # name_H = img_list_H[n]
    # name_I = img_list_I[n]

    # 获取每张图片的路径
    path_A = os.path.join(args.fold_A, name_A)
    path_B = os.path.join(args.fold_B, name_B)
    path_C = os.path.join(args.fold_C, name_C)
    path_D = os.path.join(args.fold_D, name_D)
    # path_E = os.path.join(args.fold_E, name_E)
    # path_F = os.path.join(args.fold_F, name_F)
    # path_G = os.path.join(args.fold_G, name_G)
    # path_H = os.path.join(args.fold_H, name_H)
    # path_I = os.path.join(args.fold_I, name_I)
    # 将 8 张图片的路径放入一个列表
    all_paths = [path_A, path_B, path_C, path_D]

    # 生成合并后的图片路径
    name_AB = name_A  # 图片命名格式可以根据需要调整
    path_AB = os.path.join(img_fold_AB, name_AB)

    # 检查文件是否存在
    if all(os.path.isfile(path) for path in all_paths):
        if not args.no_multiprocessing:
            pool.apply_async(resize_and_concatenate, args=(all_paths, path_AB))
        else:
            resize_and_concatenate(all_paths, path_AB)

if not args.no_multiprocessing:
    pool.close()
    pool.join()
