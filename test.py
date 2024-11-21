import cv2
import numpy as np
import matplotlib.pyplot as plt
# Step 1: 边缘检测，提取遮挡轮廓 Co 和 Cs
def extract_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Step 2: 声纳匹配和深度估计
def depth_estimation(o, Z_estimation):
    # 根据 Z_estimation 计算光学坐标系中的位置 (x_s, y_s)
    x_s = Z_estimation * (o[0] / f)
    y_s = Z_estimation * (o[1] / f)
    return x_s, y_s

# Step 3: 计算3D轮廓点 P_hat_o
def calculate_3D_contour_point(o, Z_estimation):
    x_s, y_s = depth_estimation(o, Z_estimation)
    P_hat_o = np.array([x_s, y_s, Z_estimation])
    return P_hat_o

# Step 4: 遍历 Co 上的点，获取整个光学坐标系中的三维遮挡轮廓 C
def calculate_3D_contour(Co_contours, Z_estimation):
    C = []
    for contour in Co_contours:
        for point in contour:
            o = point[0]
            P_hat_o = calculate_3D_contour_point(o, Z_estimation)
            C.append(P_hat_o)
    return np.array(C)


# Step 1: 获取光线方向 o
def get_ray_direction(camera_position, point_P):
    o = (point_P - camera_position) / np.linalg.norm(point_P - camera_position)
    return o

# Step 2: 获取局部切线 t_o
def get_tangent_vector(o):
    t_o = np.cross(o, np.array([0, 0, 1]))  # 假设轮廓在XY平面上
    t_o = t_o / np.linalg.norm(t_o)
    return t_o

# Step 3: 恢复法线 \hat{n_o}
def estimate_normal(o, t_o):
    n_hat_o = np.cross(o, t_o)
    return n_hat_o

def calculate_tangent_vector_s(s, n_hat_o):
    t_s = np.cross(s, n_hat_o)
    t_s = t_s / np.linalg.norm(t_s)
    return t_s

def estimate_normal_ss(s, t_s):
    n_hat_ss = np.cross(s, t_s)
    n_hat_ss = n_hat_ss / np.linalg.norm(n_hat_ss)
    return n_hat_ss

# 示例
s = np.array([1, 0, 0])  # 声纳图像上的点 s
#n_hat_o = np.array([0, 0, 1])  # 估计的局部法线


# 示例
camera_position = np.array([0, 0, 0])  # 光学相机位置
image = cv2.imread('1.jpg')
f = 2.12  # 举例一个焦距
Z_estimation = 7  # 举例一个深度估计值

#point_P = np.array([1, 1, 1])  # 轮廓上的点 P
Co_contours = extract_contours(image)
C = calculate_3D_contour(Co_contours, Z_estimation)

o = get_ray_direction(camera_position, C)
t_o = get_tangent_vector(o)
n_hat_o = estimate_normal(o, t_o)
t_s = calculate_tangent_vector_s(s, n_hat_o)
n_hat_ss = estimate_normal_ss(s, t_s)
print("Estimated Normal in Sonar View:", n_hat_ss)
print("Estimated Normal:", n_hat_o)
print(C)


# 假设C是一个包含三维点的numpy数组
# 示例数据

# 提取三维坐标
x, y, z = C[:, 0], C[:, 1], C[:, 2]

# 绘制3D散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
