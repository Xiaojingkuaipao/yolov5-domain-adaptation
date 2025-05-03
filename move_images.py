import os
import shutil
from PIL import Image
from tqdm import tqdm

def is_underexposed(image, low_threshold=30, ratio=0.9):
    """判断图像是否过暗，基于直方图中低亮度像素的比例"""
    hist = image.histogram()
    low_pixels = sum(hist[:low_threshold])
    total_pixels = image.size[0] * image.size[1]
    return low_pixels / total_pixels > ratio

def process_images(source_folder, under_folder):
    # 创建目标文件夹
    os.makedirs(under_folder, exist_ok=True)

    # 支持的图片扩展名
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    count = 0
    for filename in tqdm(os.listdir(source_folder), desc="Scanning Images", total=len(os.listdir(source_folder))):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
            try:
                with Image.open(file_path) as img:
                    gray_img = img.convert('L')  # 转为灰度图像
                    if is_underexposed(gray_img):
                        # 如果是移动操作，使用 shutil.move
                        # 如果只是复制，使用 shutil.copy
                        shutil.copy(file_path, os.path.join(under_folder, filename))
                        count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return count

if __name__ == "__main__":
    # 示例配置
    source_folder = './my_data/images'       # 替换为你的源图片文件夹路径
    under_folder = './my_data/under_exposed' # 过暗图片目标文件夹
    count = process_images(source_folder, under_folder)
    print(f"{count} / {len(os.listdir(source_folder))} has been copied to {under_folder}")