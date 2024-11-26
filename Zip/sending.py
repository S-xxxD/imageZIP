import cv2
import numpy as np
import subprocess
import os
import json

# 1. 选择 ROI 区域
def select_roi(image):
    roi_coords = cv2.selectROI("Select ROI", image, False, False)
    cv2.destroyWindow("Select ROI")
    return roi_coords

# 2. 白平衡和对比度增强
def process_roi(roi_image):
    def white_balance(img):
        r, g, b = cv2.split(img)
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        avg = (r_mean + g_mean + b_mean) / 3
        r = cv2.add(r, (avg - r_mean))
        g = cv2.add(g, (avg - g_mean))
        b = cv2.add(b, (avg - b_mean))
        return cv2.merge([r, g, b])

    def enhance_contrast(img):
        r, g, b = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        r_enhanced = clahe.apply(r)
        g_enhanced = clahe.apply(g)
        b_enhanced = clahe.apply(b)
        return cv2.merge([r_enhanced, g_enhanced, b_enhanced])

    def denoise_image(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    roi_image = white_balance(roi_image)
    roi_image = enhance_contrast(roi_image)
    roi_image = denoise_image(roi_image)
    return roi_image

# 3. 压缩非 ROI 区域
def compress_non_roi(image, roi_coords, output_path, rate=1):
    x, y, w, h = roi_coords
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1  # 创建 ROI 掩码
    non_roi = cv2.bitwise_and(image, image, mask=1-mask)  # 提取非 ROI 区域

    # 保存非 ROI 区域临时文件
    temp_input = "temp_input.ppm"
    cv2.imwrite(temp_input, non_roi)

    # 调用 OpenJPEG 压缩
    compress_cmd = [
        "opj_compress",
        "-i", temp_input,
        "-o", output_path,
        "-r", str(rate)
    ]
    subprocess.run(compress_cmd, check=True)
    os.remove(temp_input)

# 4. 保存 ROI 区域大小信息
def save_roi_coords(roi_coords, output_path):
    # 将 ROI 坐标和大小信息保存为 JSON 文件
    with open(output_path, 'w') as f:
        json.dump({'roi_coords': roi_coords}, f)

# 5. 主程序
def main(image_path):
    image = cv2.imread(image_path)
    roi_coords = select_roi(image)
    roi = image[roi_coords[1]:roi_coords[1]+roi_coords[3], roi_coords[0]:roi_coords[0]+roi_coords[2]]

    # 处理 ROI
    processed_roi = process_roi(roi)

    # 保存处理后的 ROI 图像
    processed_roi_path = "processed_roi.jpg"
    cv2.imwrite(processed_roi_path, processed_roi)

    # 压缩非 ROI
    compressed_path = "compressed_non_roi.j2k"
    compress_non_roi(image, roi_coords, compressed_path)

    # 保存 ROI 坐标信息
    roi_coords_path = "roi_coords.json"
    save_roi_coords(roi_coords, roi_coords_path)

    print(f"ROI Coordinates saved to {roi_coords_path}")
    print(f"Processed ROI saved to {processed_roi_path}")
    print(f"Compressed non-ROI saved to {compressed_path}")

if __name__ == "__main__":
    main("D:/PycharmProjects/Match/imageZIP/image/CO2.jpg")
