import os
import re

def extract_number(filename):
    """
    从文件名中提取数字（返回第一个连续数字）。
    如果没有数字，返回 None
    """
    m = re.search(r'\d+', filename)
    return m.group() if m else None


folder1 = "/data/cyh/Datasets/GF3_sar2opt/testB"
folder2 = "/NAS_data/yjy/GF3_combine_results"

files1 = sorted(os.listdir(folder1))
files2 = sorted(os.listdir(folder2))

for f1, f2 in zip(files1, files2):
    num1 = extract_number(f1)
    num2 = extract_number(f2)

    if num1 != num2:
        print(f"Different: {f1}  <-->  {f2}")
