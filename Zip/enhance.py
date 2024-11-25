import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt


# 1. 读取图像并手动标定ROI区域
def load_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 转换为RGB格式以便显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #mg_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_rgb


def select_roi(image):
    # 手动选择ROI区域（使用OpenCV的ROI选择工具）
    r = cv2.selectROI("Select ROI", image)
    # 检查选择的区域是否有效
    if r[2] == 0 or r[3] == 0:
        print("No valid ROI selected!")
        return None, None
    # 按下'Enter'后选定区域
    roi_image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    cv2.destroyWindow("Select ROI")
    return r, roi_image  # 返回ROI区域坐标和提取的ROI图像

def show_image(image, title="Image"):
    # 打印图像的形状和类型，帮助调试
    print(f"Image Shape: {image.shape}")
    print(f"Image Type: {image.dtype}")

    # 检查图像类型并转换为适合显示的格式
    if len(image.shape) == 2:  # 灰度图
        plt.imshow(image, cmap='gray')
        print("222")
    elif len(image.shape) == 3:  # 彩色图
        plt.imshow(image)
    else:
        raise ValueError("Invalid image shape")

    plt.title(title)
    plt.axis('off')
    plt.show()


# 白平衡修正（简单的平均法，或者使用更复杂的白平衡算法）
def white_balance(image):
    # 计算RGB三个通道的平均值
    r, g, b = cv2.split(image)
    r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)

    # 计算整体平均值
    avg = (r_mean + g_mean + b_mean) / 3

    # 调整各通道的亮度，达到色彩平衡
    r = cv2.add(r, (avg - r_mean))
    g = cv2.add(g, (avg - g_mean))
    b = cv2.add(b, (avg - b_mean))

    # 合并回RGB图像
    balanced_image = cv2.merge([r, g, b])
    return balanced_image


# 对比度增强（CLAHE）
def enhance_contrast(image):
    # 分别对RGB三个通道进行CLAHE增强
    r, g, b = cv2.split(image)

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

# 对比度和亮度调整
def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    # alpha > 1.0 增加对比度，alpha < 1.0 减少对比度
    # beta 用于亮度调整，增加beta增加亮度，减少beta降低亮度
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def compress_non_roi(image, roi_coords, compression_factor=0.5, quality=10):
    x, y, w, h = roi_coords
    height, width, _ = image.shape

    # 保留ROI区域
    roi = image[y:y + h, x:x + w]

    # 提取非ROI区域
    mask = np.ones((height, width), dtype=np.uint8)
    mask[y:y + h, x:x + w] = 0
    background = cv2.bitwise_and(image, image, mask=mask)

    # JPEG压缩非ROI区域
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed_background = cv2.imencode('.jpg', background, encode_param)
    decompressed_background = cv2.imdecode(compressed_background, cv2.IMREAD_COLOR)

    # 平滑处理
    decompressed_background = cv2.GaussianBlur(decompressed_background, (15, 15), 0)

    # 替换ROI区域
    decompressed_background[y:y + h, x:x + w] = roi

    return decompressed_background


# 主程序
def main(image_path):
    # 1. 加载图像
    image = load_image(image_path)
    show_image(image, "Original Image")

    # 预处理：颜色校正
    image = white_balance(image)

    # 预处理：增强对比度
    image = enhance_contrast(image)

    # 预处理：去噪
    image = denoise_image(image)

    show_image(image, "Enhanced Image")

    # 2. 手动标定ROI区域
    roi_coords, roi_image = select_roi(image)

    # 如果没有选择有效的ROI，则终止程序
    if roi_coords is None:
        print("Exiting program due to invalid ROI!")
        return

    compressed_image = compress_non_roi(image, roi_coords, compression_factor=0.2, quality=10)
    show_image(compressed_image, "Compressed Non-ROI with JPEG")


if __name__ == "__main__":
    image_path = 'D:/PycharmProjects/Match/imageZIP/image/CO2.jpg'  # 替换为你的图像路径
    main(image_path)
