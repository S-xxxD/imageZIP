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


# 2. 使用小波变换进行多分辨率分解
def wavelet_transform(image):
    # 将每个通道单独进行小波变换
    LL = []
    LH = []
    HL = []
    HH = []

    # 对每个通道进行小波变换
    for i in range(3):  # 处理RGB三个通道
        channel = image[:, :, i]
        coeffs2 = pywt.dwt2(channel, 'haar')  # 2D离散小波变换
        LL_ch, (LH_ch, HL_ch, HH_ch) = coeffs2  # LL:低频, LH:水平高频, HL:垂直高频, HH:对角高频

        # 将每个通道的小波系数存储
        LL.append(LL_ch)
        LH.append(LH_ch)
        HL.append(HL_ch)
        HH.append(HH_ch)

    return LL, LH, HL, HH


# 3. 对ROI区域进行增强，ROI外区域进行压缩
def process_image(LL, LH, HL, HH, roi_coords):
    # 获取ROI区域的坐标
    x, y, w, h = roi_coords

    for i in range(3):  # 对每个通道处理
        # 提取每个通道的小波系数
        roi_LL = LL[i][y:y + h, x:x + w]
        roi_LH = LH[i][y:y + h, x:x + w]
        roi_HL = HL[i][y:y + h, x:x + w]
        roi_HH = HH[i][y:y + h, x:x + w]

        # 1. 增强ROI区域（提高细节）
        roi_LL *= 1.5  # 增强低频区域
        roi_LH *= 1.4  # 增强水平高频区域
        roi_HL *= 1.4  # 增强垂直高频区域
        roi_HH *= 1.4  # 增强对角高频区域

        # 将增强后的ROI区域放回原小波系数
        LL[i][y:y + h, x:x + w] = roi_LL
        LH[i][y:y + h, x:x + w] = roi_LH
        HL[i][y:y + h, x:x + w] = roi_HL
        HH[i][y:y + h, x:x + w] = roi_HH

        # 2. 对ROI外区域进行压缩（降低细节）
        # 对ROI外的区域进行小波系数的压缩（例如，降低它们的幅度）
        # ROI外区域坐标
        non_roi_y = [0, y, y + h, LL[i].shape[0]]
        non_roi_x = [0, x, x + w, LL[i].shape[1]]

        # 处理外区域的小波系数：通过减少幅度进行压缩
        for j in range(0, len(non_roi_y)-1):
            for k in range(0, len(non_roi_x)-1):
                # 对每个块进行压缩操作，减少系数值
                LL[i][non_roi_y[j]:non_roi_y[j+1], non_roi_x[k]:non_roi_x[k+1]] *= 0.5
                LH[i][non_roi_y[j]:non_roi_y[j+1], non_roi_x[k]:non_roi_x[k+1]] *= 0.5
                HL[i][non_roi_y[j]:non_roi_y[j+1], non_roi_x[k]:non_roi_x[k+1]] *= 0.5
                HH[i][non_roi_y[j]:non_roi_y[j+1], non_roi_x[k]:non_roi_x[k+1]] *= 0.5

        # 限制增强后的小波系数范围，防止亮度过高（避免过曝）
        LL[i] = np.clip(LL[i], 0, 255)
        LH[i] = np.clip(LH[i], 0, 255)
        HL[i] = np.clip(HL[i], 0, 255)
        HH[i] = np.clip(HH[i], 0, 255)

    return LL, LH, HL, HH


# 4. 使用分层编码和比特分配
def layer_encoding(LL, LH, HL, HH):
    encoded_layers = []

    for i in range(3):  # 对每个通道
        coeffs_encoded = np.round(LL[i] / 2)  # 简单压缩
        encoded_layers.append(coeffs_encoded)

    return encoded_layers


# 5. 渐进式编码和逐层解码
def progressive_encoding(encoded_layers):
    progressive_image = np.zeros_like(encoded_layers[0], dtype=np.float32)  # 初始化为与编码层相同的形状

    for layer in encoded_layers:
        progressive_image += layer  # 简单累加模拟

    # 确保 progressive_image 为适合显示的形状和类型
    progressive_image = np.clip(progressive_image, 0, 255)  # 限制值的范围为 0 到 255
    progressive_image = progressive_image.astype(np.uint8)  # 确保是整数类型

    # 如果图像是灰度图像，需要转换为三通道的图像才能显示
    if len(progressive_image.shape) == 2:  # 灰度图像
        progressive_image = cv2.cvtColor(progressive_image, cv2.COLOR_GRAY2RGB)
        print("gray")

    return progressive_image


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

    # 3. 小波变换分解
    LL, LH, HL, HH = wavelet_transform(image)

    # 4. 对ROI区域进行增强，ROI外区域进行压缩
    LL, LH, HL, HH = process_image(LL, LH, HL, HH, roi_coords)

    # 5. 分层编码（模拟）
    encoded_layers = layer_encoding(LL, LH, HL, HH)

    # 6. 渐进式编码与逐层解码（模拟）
    progressive_image = progressive_encoding(encoded_layers)

    progressive_image = adjust_brightness_contrast(progressive_image, alpha=0.9, beta=-10)
    show_image(progressive_image, "Progressive Image (After Encoding)")


if __name__ == "__main__":
    image_path = 'D:/PycharmProjects/Match/image/Co2.jpg'  # 替换为你的图像路径
    main(image_path)
