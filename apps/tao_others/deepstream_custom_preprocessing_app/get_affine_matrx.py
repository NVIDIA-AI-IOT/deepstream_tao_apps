import numpy as np
import math
np.set_printoptions(suppress=True)


Input_Width = 1920
Input_Height = 1080
Output_Width = 960
Output_Height = 544
alpha = np.pi/4


T = np.array([
    [1, 0, -Input_Width * 0.5],
    [0, 1, -Input_Height * 0.5],
    [0, 0, 1]
])

R = np.array([
    [math.cos(alpha), -math.sin(alpha), 0],
    [math.sin(alpha), math.cos(alpha), 0],
    [0, 0, 1]
])

w_scale = Output_Width/Input_Width
h_scale = Output_Height/Input_Height
sx = sy = min(w_scale, h_scale)

S = np.array([
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1]
])

invT = np.array([
    [1, 0, Output_Width/2],
    [0, 1, Output_Height/2],
    [0, 0, 1]
])

M = invT @ S @ R @ T

print(M)
 