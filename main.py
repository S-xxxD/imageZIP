import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, solve, lambdify
import cv2

image_path = 'image/Co1.jpg'
image = cv2.imread(image_path)
image_width = image.shape[1]
image_height = image.shape[0]

theta = np.deg2rad(20)
Re = 6.8
tx = 0.06
ty = 0.0712
tz = 0.334133
# tx = 0
# ty = 0
# tz = 0.334133
x_offset, y_offset = 400, 799  # 声纳图像的原点坐标
t = np.array([[tx], [ty], [tz]])
norm_t = np.linalg.norm(t)

r1 = np.array([[1, 0, 0]])
r2 = np.array([[0, 0, 1]])
r3 = np.array([[0, -1, 0]])
R = np.array([r1[0], r2[0], r3[0]])
print(R @ R.T)

x_pixels = np.linspace(0, image_width - 1, image_width)
y_pixels = np.linspace(0, image_height - 1, image_height)
x = x_pixels
y = y_pixels

def calculate_U(theta, ty, tx, norm_t, Re, r1, r2):

    term1 = (np.tan(theta) * ty - tx)**2 * np.eye(3)
    term2 = (norm_t ** 2 - Re ** 2) * ((r1 - np.tan(theta) * r2).T @ (r1 - np.tan(theta) * r2))
    term3 = (np.tan(theta) * ty - tx) * ((r1 - np.tan(theta) * r2).T @ (t.T @ R) + (R.T @ t) @ (r1 - np.tan(theta) * r2))

    U = term1 + term2 + term3

    return U

def quadratic_form(A, x_value):
    a, b, c, d, e, f = A[0][0], A[0][1], A[0][2], A[1][1], A[1][2], A[2][2]
    x, y, z = symbols('x y z')
    x_value = x_value - 400
    # 二次型方程
    quadratic_equation = a * x ** 2 + d * y ** 2 + f * 1 ** 2 + 2 * b * x * y + 2 * e * y * 1 + 2 * c * x * 1
    # 解出 y 的表达式

    y_expression = solve(quadratic_equation, y)[1]
    y_function = lambdify(x, y_expression, 'numpy')
    # 计算每个 y 对应的 x
    y_values = y_function(x_value)
    print(y_expression)
    return y_values

U = calculate_U(theta,  ty, tx, norm_t, Re, r1, r2)
print(U)

K = quadratic_form(U, x)

print(K)

plt.imshow(image)
plt.plot(x, K, label='Upper Branch', color='red')

plt.title('Hyperbolic Lines on Image')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.ylim(image_height, 0)
plt.xlim(0, image_width)
plt.legend()
plt.show()
