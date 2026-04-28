import os
import shutil

def extract_pred_edge_files(src_dir, dst_dir):
    """
    遍历 src_dir 下的所有文件，
    将文件名中包含 'pred_edge' 的文件复制到 dst_dir。
    """
    # 若目标文件夹不存在，则创建
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    count = 0
    for root, _, files in os.walk(src_dir):
        for file in files:
            if "_fake_B" in file:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_dir, file)

                # 如果文件名重复，可以选择加上上级文件夹名区分
                # dst_path = os.path.join(dst_dir, f"{os.path.basename(root)}_{file}")

                shutil.copy2(src_path, dst_path)
                count += 1
                print(f"✅ Copied: {src_path} -> {dst_path}")

    print(f"\n共提取 {count} 个包含 'pred_edge' 的文件。")


if __name__ == "__main__":
    # ===== 修改这里为你的路径 =====
    source_folder = "/NAS_data/yjy/GF3_combine_results"   # 原始文件夹路径
    target_folder = "/NAS_data/yjy/GF3_combine_results_1"  # 要保存的目标文件夹

    extract_pred_edge_files(source_folder, target_folder)
