import cv2
import os
import numpy as np
from tqdm import tqdm


def apply_gamma_correction(image, gamma=1.0):
    """
    Apply Gamma Correction to the input image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def find_best_gamma(patch):
    """
    Find the best gamma value that maximizes the variance of the patch.
    """
    best_gamma = 1.0
    max_variance = 0
    gammas = [0.5 + i * 0.1 for i in range(16)]  # Gamma values from 0.5 to 2.0
    for gamma in gammas:
        corrected = apply_gamma_correction(patch, gamma)
        variance = np.var(corrected)
        if variance > max_variance:
            max_variance = variance
            best_gamma = gamma
    return best_gamma

def patch_augmentation(image, patch_size=100):
    """
    Apply adaptive patch augmentation using Gamma correction.
    """
    h, w = image.shape
    augmented = np.zeros_like(image)
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            # Extract patch
            patch = image[y:y+patch_size, x:x+patch_size]
            # Find best gamma
            best_gamma = find_best_gamma(patch)
            # Apply gamma correction
            corrected_patch = apply_gamma_correction(patch, best_gamma)
            # Place back to the augmented image
            augmented[y:y+patch_size, x:x+patch_size] = corrected_patch
    return augmented

def global_equalization(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def process_image(image_path, output_path):
    """
    Process a single image using APAGE method.
    """
    image = cv2.imread(image_path, 0)  # Read as grayscale
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    # Step 1: Patch Augmentation
    augmented = patch_augmentation(image)
    # Step 2: Global Equalization
    enhanced = global_equalization(augmented)
    # Save the result
    cv2.imwrite(output_path, enhanced)


def batch_process(input_dir, output_dir):
    """
    批量处理图像并显示进度条
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有支持格式的文件
    valid_extensions = (".png", ".jpg", ".jpeg")
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    # 使用 tqdm 添加进度条
    for filename in tqdm(file_list, desc="Processing Images", total=len(file_list)):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        process_image(input_path, output_path)

if __name__ == "__main__":
    input_dir = "./my_data/under_exposed"
    output_dir = "./my_data/enhanced_images"
    batch_process(input_dir, output_dir)