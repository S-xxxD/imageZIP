import numpy as np
import matplotlib.pyplot as plt
import cv2

# 替换为你的图像路径
image_path = 'image/Co1.jpg'

# 读取图像
image = cv2.imread(image_path)

# 图像的实际大小
image_width = image.shape[1]
image_height = image.shape[0]
print(image.shape)
print(image_width)
print(image_height)
# 参数
tx = 8

Re = 6.7
theta = np.deg2rad(4.1)

# 生成 x 的范围
x_pixels = np.linspace(0, image_width - 1, image_width)
x = x_pixels

# 计算对应的 y
# #expression_inside_sqrt = np.maximum(0, ((x - np.tan(theta))**2) / ((tx / Re)**2))  # 避免负数
# expression_inside_sqrt = ((x - np.tan(theta))**2) / ((tx / Re)**2)
# #y_positive = np.sqrt(1 + np.tan(theta)**2) * np.sqrt(expression_inside_sqrt)  # 上半部分
# y_positive = np.sqrt((expression_inside_sqrt)-(1 + np.tan(theta)**2))  # 上半部分
# y_negative = -y_positive  # 下半部分
expression_inside_sqrt = np.maximum(0, ((x - np.tan(theta)) / (tx / Re))**2 - 1 - np.tan(theta)**2)
y_positive = np.sqrt(expression_inside_sqrt)
print(expression_inside_sqrt)
y_negative = -y_positive
y_positive = np.real(y_positive)
y_negative = np.real(y_negative)
# 绘制图像和双曲线
plt.imshow(image)
plt.plot(x_pixels, y_positive, label='Upper Branch', color='red')
#plt.plot(x_pixels, y_negative, label='Lower Branch', color='blue')

# 设置标题和标签
plt.title('Hyperbolic Lines on Image')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.ylim(800, 0)
plt.xlim(0, image_width)
# 显示图例
plt.legend()

# 显示图形
plt.show()
