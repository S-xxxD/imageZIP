import cv2
import subprocess
import os
import json


# 1. 读取ROI区域坐标
def load_roi_coords(roi_coords_path):
    with open(roi_coords_path, 'r') as f:
        data = json.load(f)
    return data['roi_coords']


# 2. 解压非 ROI 区域
def decompress_non_roi(compressed_path, output_path):
    decompress_cmd = [
        "opj_decompress",
        "-i", compressed_path,
        "-o", output_path
    ]
    subprocess.run(decompress_cmd, check=True)


# 3. 合并处理后的 ROI 区域和解压后的非 ROI 区域
def combine_images(non_roi_image, roi_coords, processed_roi):
    x, y, w, h = roi_coords
    combined_image = non_roi_image.copy()
    combined_image[y:y + h, x:x + w] = processed_roi
    return combined_image


# 4. 主程序
def main(compressed_path, roi_path, roi_coords_path, output_path):
    # 读取 ROI 坐标信息
    roi_coords = load_roi_coords(roi_coords_path)

    # 解压非ROI区域
    temp_output = "temp_output.ppm"
    decompress_non_roi(compressed_path, temp_output)

    # 读取解压后的非ROI区域图像
    non_roi_image = cv2.imread(temp_output)
    os.remove(temp_output)  # 删除临时解压文件

    # 读取处理后的ROI区域
    processed_roi = cv2.imread(roi_path)

    # 合并图像
    combined_image = combine_images(non_roi_image, roi_coords, processed_roi)

    # 保存最终合成的图像
    cv2.imwrite(output_path, combined_image)
    print(f"Final image saved to: {output_path}")


if __name__ == "__main__":
    compressed_path = "compressed_non_roi.j2k"  # 接收的压缩后的非ROI区域路径
    roi_path = "processed_roi.jpg"  # 接收的处理后的ROI区域路径
    roi_coords_path = "roi_coords.json"  # 接收的ROI区域坐标文件路径
    output_path = "final_image.jpg"  # 输出最终合成图像的路径

    main(compressed_path, roi_path, roi_coords_path, output_path)
