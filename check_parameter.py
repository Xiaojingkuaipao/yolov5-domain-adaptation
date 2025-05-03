"""
此文件用于统计过暗以及过曝图片的参数
使用方法：
    挑选你认为过暗的图片放到under_exposed，把认为过亮的一些图片放到over_exposed文件夹中
    运行此脚本 python check_parameter.py
判断指标：
    此脚本使用像素直方图中的统计信息判断是否过暗或者过曝
    包括像素取值的阈值以及比例
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] # 设置字体
plt.rcParams["axes.unicode_minus"]=False  # 该语句解决图像中的“-”负号的乱码问题

def get_histogram_stats(folder_path):
    """获取文件夹中所有图片的灰度直方图统计信息"""
    all_hist = []
    all_low_ratio = []
    all_high_ratio = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                with Image.open(file_path) as img:
                    gray_img = img.convert('L')
                    hist = gray_img.histogram()
                    total_pixels = gray_img.size[0] * gray_img.size[1]

                    # 计算低亮度像素比例（假设 low_threshold = 50）
                    low_threshold = 50
                    low_pixels = sum(hist[:low_threshold])
                    low_ratio = low_pixels / total_pixels
                    all_low_ratio.append(low_ratio)

                    # 计算高亮度像素比例（假设 high_threshold = 200）
                    high_threshold = 200
                    high_pixels = sum(hist[high_threshold:])
                    high_ratio = high_pixels / total_pixels
                    all_high_ratio.append(high_ratio)

                    all_hist.append(hist)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return np.array(all_hist), np.array(all_low_ratio), np.array(all_high_ratio)

def analyze_thresholds_and_ratio(all_hist, all_low_ratio, all_high_ratio):
    """分析直方图数据，推荐合适的 low_threshold, high_threshold 和 ratio"""

    # 灰度直方图平均值
    avg_hist = np.mean(all_hist, axis=0)

    # 推荐的 low_threshold：找到前 80 个灰度值中，累计占比超过 80% 的位置
    cumsum = np.cumsum(avg_hist[:80]) / np.sum(avg_hist[:80])
    low_threshold = np.argmax(cumsum > 0.8)

    # 推荐的 high_threshold：从后往前找，累计占比超过 80%
    cumsum_high = np.cumsum(avg_hist[-80:][::-1]) / np.sum(avg_hist[-80:])
    high_threshold = 255 - np.argmax(cumsum_high > 0.8)

    # 推荐的 ratio：取所有“过暗”图片的 low_ratio 的平均值
    avg_low_ratio = np.mean(all_low_ratio)
    avg_high_ratio = np.mean(all_high_ratio)

    print(f"推荐的低亮度阈值 (low_threshold): {low_threshold}")
    print(f"推荐的高亮度阈值 (high_threshold): {high_threshold}")
    print(f"推荐的低亮度比例 (ratio): {avg_low_ratio:.2f}")
    print(f"推荐的高亮度比例 (ratio): {avg_high_ratio:.2f}")

    return low_threshold, high_threshold, avg_low_ratio, avg_high_ratio

def plot_histogram(all_hist, title="灰度直方图"):
    """绘制平均灰度直方图"""
    avg_hist = np.mean(all_hist, axis=0)
    plt.figure(figsize=(10, 4))
    plt.bar(range(256), avg_hist, width=1, color='gray')
    plt.title(title)
    plt.xlabel("灰度值")
    plt.ylabel("像素数量")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    under_folder = './my_data/under_exposed'
    over_folder = './my_data/over_exposed'

    # 获取过暗图片的直方图统计
    under_hist, under_low_ratio, _ = get_histogram_stats(under_folder)
    plot_histogram(under_hist, "过暗图片的平均灰度直方图")

    # 获取过曝图片的直方图统计
    over_hist, _, over_high_ratio = get_histogram_stats(over_folder)
    plot_histogram(over_hist, "过曝图片的平均灰度直方图")

    # 推荐合适的阈值和比例
    low_threshold, high_threshold, avg_low_ratio, avg_high_ratio = analyze_thresholds_and_ratio(
        under_hist, under_low_ratio, over_high_ratio
    )