import numpy as np
import cv2

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

def main(image_path):
    image = cv2.imread(image_path)
    processed_roi = process_roi(image)
    # 保存处理后的 ROI 图像
    processed_roi_path = "processed_roi.jpg"
    cv2.imwrite(processed_roi_path, processed_roi)
if __name__ == "__main__":
    main("D:/PycharmProjects/Match/imageZIP/image/CO2.jpg")