import os
import shutil

# ==========================
# 1️⃣ 路径配置
# ==========================
sen12_scene_root = r"/NAS_data/yjy/DDBM_S2O_canny_results/S2O_scene_v2/result_200000"
sen1_2_bit_root = r"/data/yjy_data/SEN1-2_BIT"
output_root = r"/NAS_data/yjy/SEN1-2-BIT_matched/Test"

# ==========================
# 2️⃣ 获取 SEN12_Scene 下所有文件名
# ==========================
scene_filenames = set()
for root, dirs, files in os.walk(sen12_scene_root):
    for f in files:
        scene_filenames.add(f)

print(f"✅ 已读取 {len(scene_filenames)} 个目标文件用于匹配。")

# ==========================
# 3️⃣ 遍历 SEN1-2_BIT 文件夹，查找匹配文件
# ==========================
matched_count = 0
dst_dir = os.path.join(output_root, "pred_pix2pix")
os.makedirs(dst_dir, exist_ok=True)  # 统一放在 All 下

for season in os.listdir(sen1_2_bit_root):
    season_path = os.path.join(sen1_2_bit_root, season)
    if not os.path.isdir(season_path):
        continue

    for category in os.listdir(season_path):  # Farmland, Forest, etc.
        category_path = os.path.join(season_path, category)
        if not os.path.isdir(category_path):
            continue

        for root, _, files in os.walk(category_path):
            for f in files:
                if f in scene_filenames:
                    src_path = os.path.join(root, f)

                    # 文件名加类别后缀
                    name, ext = os.path.splitext(f)
                    new_name = f"{name}_{category}{ext}"
                    dst_path = os.path.join(dst_dir, new_name)

                    shutil.copy2(src_path, dst_path)
                    matched_count += 1

print(f"✅ 匹配完成，共复制 {matched_count} 个文件到 {dst_dir}")
# import os
# import re
#
# # ===============================
# # 1️⃣ 配置路径
# # ===============================
# folderA = r"/NAS_data/yjy/SEN1-2-BIT_matched/Test/B"
# folderB = r"/data/hjf/Dataset/SEN12_Scene/testB"
#
# # ===============================
# # 2️⃣ 定义函数：去掉类别后缀
# # ===============================
# def remove_category_suffix(filename):
#     """
#     去掉文件名中最后一个下划线后的类别后缀
#     例如 '001_Farmland.png' -> '001'
#     """
#     name, _ = os.path.splitext(filename)
#     base = re.sub(r"_[^_]+$", "", name)  # 去掉最后一个下划线及其后内容
#     return base
#
# # ===============================
# # 3️⃣ 获取文件名集合
# # ===============================
# # B 的文件名集合（不带扩展名）
# B_names = {os.path.splitext(f)[0] for f in os.listdir(folderB)
#            if os.path.isfile(os.path.join(folderB, f))}
#
# # A 的文件列表，用于逐个比对
# A_files = [f for f in os.listdir(folderA)
#            if os.path.isfile(os.path.join(folderA, f))]
#
# # ===============================
# # 4️⃣ 匹配逻辑
# # ===============================
# not_in_B = []
# for f in A_files:
#     base = remove_category_suffix(f)
#     if base not in B_names:
#         not_in_B.append(f)
# # ===============================
# # 5️⃣ 输出结果
# # ===============================
# print("✅ 以下文件在 A 中存在，但在 B 中不存在（去除类别后缀后）：")
# for f in not_in_B:
#     print("  ", f)
#
# print(f"\n📊 总计：{len(not_in_B)} 个文件未匹配。")
