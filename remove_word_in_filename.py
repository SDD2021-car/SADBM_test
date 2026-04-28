import os
import shutil

def remove_fake_B_from_filenames(directory):
    # 遍历指定目录
    for filename in os.listdir(directory):
        # 检查文件名中是否包含'_fake_B'
        if '_fake_B' in filename:
            # 构造原始文件的完整路径
            old_file_path = os.path.join(directory, filename)
            # 构造新的文件名，去掉'_fake_B'
            new_filename = filename.replace('_fake_B', '')
            # 构造新文件的完整路径
            new_file_path = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{old_file_path}' to '{new_file_path}'")

# 指定要处理的目录
directory = '/NAS_data/cyh/DOGAN_all_determined_experiment/results/DOGAN_GF3_5.26_remake1/images'

# 调用函数
remove_fake_B_from_filenames(directory)
a.pop()