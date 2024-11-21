# 光-声对应
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = 'image/son11.jpg'
image = cv2.imread(image_path)
image_width = image.shape[1]
image_height = image.shape[0]

# 相机的内参数矩阵K
K =[[527.8346, 0, 654.6259],
    [0, 527.6546, 367.7943],
    [0, 0, 1]]
x_offset, y_offset = 423, 458  # 声纳图像的原点坐标

x_pix = 359  # 光学图像像素坐标

k = (400 - x_pix) / K[0][0]
x = np.linspace(0, image_width - 1, image_width)
y = 1/k * (x - x_offset) + y_offset

plt.imshow(image)
plt.plot(x, y, label='Upper Branch', color='red')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.ylim(image_height, 0)
plt.legend()
plt.show()
