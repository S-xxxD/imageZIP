import cv2
import numpy as np

# 确保您的图像路径是正确的
image_path = 'img1.jpg'

# 读取图像
image = cv2.imread(image_path)

# 预处理: 转换为灰度图并进行高斯模糊去噪
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

x = 476
y = 267
width = 200
height = 70
# 定义感兴趣区域的坐标或边界框
roi_coordinates = (x, y, width, height)

# 提取感兴趣区域
roi = image[y:y+height, x:x+width]

# 转为灰度图
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


# 边缘检测
edges = cv2.Canny(gray_roi, 100, 200)

# 寻找物体边界
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 添加感兴趣区域的左上角坐标到每个轮廓点
for contour in contours:
    for point in contour:
        point[0][0] += x
        point[0][1] += y

# 绘制边界
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # 在原图上绘制边界

# 显示结果
cv2.imshow('Edge Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()