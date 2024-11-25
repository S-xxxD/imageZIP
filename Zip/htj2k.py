import cv2
import numpy as np
import subprocess
import os


# 1. 选择 ROI 区域
def select_roi(image):
    roi_coords = cv2.selectROI("Select ROI", image, False, False)
    cv2.destroyWindow("Select ROI")
    return roi_coords


# 2. 白平衡和对比度增强
def process_roi(roi_image):
    # 白平衡
    def white_balance(img):
        r, g, b = cv2.split(img)
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        avg = (r_mean + g_mean + b_mean) / 3
        r = cv2.add(r, (avg - r_mean))
        g = cv2.add(g, (avg - g_mean))
        b = cv2.add(b, (avg - b_mean))
        return cv2.merge([r, g, b])

    # 对比度增强
    def enhance_contrast(img):
        # 分别对RGB三个通道进行CLAHE增强
        r, g, b = cv2.split(img)

        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # 分别对每个通道应用CLAHE
        r_enhanced = clahe.apply(r)
        g_enhanced = clahe.apply(g)
        b_enhanced = clahe.apply(b)

        # 合并增强后的三个通道
        enhanced_image = cv2.merge([r_enhanced, g_enhanced, b_enhanced])

        return enhanced_image

    # 去噪（高斯滤波）
    def denoise_image(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    roi_image = white_balance(roi_image)
    roi_image = enhance_contrast(roi_image)
    roi_image = denoise_image(roi_image)
    return roi_image


# 3. 压缩非 ROI 区域
def compress_non_roi(image, roi_coords, output_path, rate=20):
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


# 4. 合并图像
def combine_images(image, roi_coords, processed_roi, compressed_path):
    x, y, w, h = roi_coords

    # 解压非 ROI 压缩文件
    temp_output = "temp_output.ppm"
    decompress_cmd = [
        "opj_decompress",
        "-i", compressed_path,
        "-o", temp_output
    ]
    subprocess.run(decompress_cmd, check=True)
    decompressed_non_roi = cv2.imread(temp_output)
    os.remove(temp_output)

    # 合并 ROI 和非 ROI 区域
    combined_image = decompressed_non_roi.copy()
    combined_image[y:y+h, x:x+w] = processed_roi
    return combined_image


# 主程序
def main(image_path):
    image = cv2.imread(image_path)
    roi_coords = select_roi(image)
    roi = image[roi_coords[1]:roi_coords[1]+roi_coords[3], roi_coords[0]:roi_coords[0]+roi_coords[2]]

    # 处理 ROI
    processed_roi = process_roi(roi)

    # 压缩非 ROI
    compressed_path = "compressed.j2k"
    compress_non_roi(image, roi_coords, compressed_path)

    # 合并图像
    combined_image = combine_images(image, roi_coords, processed_roi, compressed_path)
    cv2.imshow("Final Image", combined_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main("D:/PycharmProjects/Match/imageZIP/image/CO3.jpg")