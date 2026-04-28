import os
from PIL import Image

def check_images(root_dir):
    corrupted = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(root, file)

            # 仅检查常见图像格式
            ext = file.lower().split('.')[-1]
            if ext not in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp']:
                continue

            try:
                img = Image.open(path)
                img.verify()  # 只检查，不加载像素
            except Exception as e:
                corrupted.append(path)
                print(f"[CORRUPTED] {path} | Error: {e}")

    print("\n==== 检查完成 ====")
    print(f"总共损坏图像数量: {len(corrupted)}")
    return corrupted

if __name__ == "__main__":
    folder = "/NAS_data/yjy/SEN1-2-BIT_matched/Test/fakeB"   # 修改为你的图像文件夹路径
    corrupted_list = check_images(folder)
