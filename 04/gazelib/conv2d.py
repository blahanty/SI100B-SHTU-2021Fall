import numpy as np



# kernels
sobel_hrztl = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sobel_vtcl = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

gaussian_knl = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16.

