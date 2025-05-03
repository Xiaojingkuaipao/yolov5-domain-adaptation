import os
import cv2
from tqdm import tqdm

def convert_grayscale_to_rgb(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有文件列表
    file_list = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]
    total_files = len(file_list)

    # 遍历文件并处理
    for idx, filename in enumerate(tqdm(file_list, desc="Processing Images", total=total_files)):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # 读取图像（默认为BGR格式）
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to read image: {filename}")
                continue

            # 检查是否为灰度图（单通道）
            if len(image.shape) == 2:
                # 转换为RGB格式
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # 如果是BGR格式，转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 其他格式（如RGBA）可按需处理
                print(f"Unsupported format for {filename}, skipping.")
                continue

            # 保存为JPEG格式
            cv2.imwrite(output_path, image)
            # print(f"Converted: {filename} -> {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = "./my_data/images"
    output_folder = "./my_data/rgb_images"
    convert_grayscale_to_rgb(input_folder, output_folder)