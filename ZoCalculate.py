import numpy as np

# 已知参数
# 相机的内参数矩阵K
K =[[527.8346, 0, 654.6259],
    [0, 527.6546, 367.7943],
    [0, 0, 1]]
x_pix = 462
y_pix = 63
x = x_pix
y = y_pix
p = np.array([x, y, 1])  # 像素坐标
A = np.linalg.inv(K) @ p
print(A)
p_norm = np.sqrt((x * 1)**2 + (y * 1)**2)
A_norm = np.linalg.norm(A)
r1 = np.array([[1, 0, 0]])
r2 = np.array([[0, 0, 1]])
r3 = np.array([[0, -1, 0]])
# r1 = np.array([[0.1302, -0.9792, -0.1557]])
# r2 = np.array([[-0.7968, -0.1968, 0.5712]])
# r3 = np.array([[-0.5900, 0.0497, -0.8059]])
R = np.array([r1[0], r2[0], r3[0]])
tx = 60     # 毫米
ty = 71.2  # 毫米
tz = 334  # 毫米
# tx = 0
# ty = 0
# tz = 0

t = np.array([[tx], [ty], [tz]])
Re = 6900 # 已知距离R
theta = np.deg2rad(5.5)  # 已知偏转角

# 计算方程系数
a = (A_norm)**2
b = (2) * t.T @ R @ np.linalg.inv(K) @ p
c = np.linalg.norm(t)**2 - Re**2

# 计算二次方程的根
delta = b**2 - 4 * a * c

if delta >= 0:
    Zo_1 = (-b + np.sqrt(delta)) / (2 * a)
    Zo_2 = (-b - np.sqrt(delta)) / (2 * a)
    print("Possible depths:", Zo_1, Zo_2)
else:
    print("No real roots for Z_o.")

# 选择实际情况中合适的解
# 根据具体问题，选择 Zo_1 或 Zo_2 作为深度值
