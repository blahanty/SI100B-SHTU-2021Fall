import numpy as np

def yaw_pitch2vec(yp):
    yaw, pitch = yp[0], yp[1]
    y = -np.sin(pitch)
    proj_zox = np.cos(pitch)

    z = -np.cos(yaw) * proj_zox
    x = -np.sin(yaw) * proj_zox

    return np.array([x, y, z])
