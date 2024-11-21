# 声-光对应
import numpy as np
import matplotlib.pyplot as plt
import cv2
image_path = 'image/Co2.jpg'
image = cv2.imread(image_path)
image_width = image.shape[1]
image_height = image.shape[0]

# 相机的内参数矩阵K
K =[[527.8346, 0, 654.6259],
    [0, 527.6546, 367.7943],
    [0, 0, 1]]
x_offset, y_offset = 423, 458  # 声纳图像的原点坐标

x, y= 488, 326 # 声纳图像的像素坐标
xs = x - x_offset
ys = y - y_offset

yo = np.linspace(0, image_height - 1, image_height)
xo = 400 - K[0][0] * xs/ys + 0 * yo
print(xo)

plt.imshow(image)
plt.plot(xo, yo, label='Upper Branch', color='red')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.ylim(image_height, 0)
plt.legend()
plt.show()
