import numpy as np
import os
import pandas as pd

sanity_Y_est = np.array(
    [
        [0, 1, 2],
        [4, 5, 1]
    ]
)

sanity_Y_gt = np.array(
    [
        [-4, 2, 2],
        [3, 2, 6]
    ]
)


sanity_X_train = np.array(
    [
        [0, 1, 2],
        [4, 5, 1]
    ]
)

sanity_Y_train = np.array(
    [
        [4, 1],
        [2, 1]
    ]
)

sanity_W = np.array(
    [[0.2, -0.2],
    [0.2, -0.2]]
)

sanity_b = np.array(
    [[0.1],
     [-0.1]]
)

sanity_dZ = np.array(
    [[0.5, -0.5, 0.2],
     [0.5, -0.5, -0.4]]
)

def assert_eq(val_a, val_b, epilsion=1e-3):
    """

    :param val_a:
    :param val_b:
    :param epilsion:
    :return:
    """
    assert abs(val_a - val_b) < epilsion

def assert_eq_np(val_a, val_b, epilsion=1e-3):
    """

    :param val_a:
    :param val_b:
    :param epilsion:
    :return:
    """
    assert (np.abs(val_a - val_b) < epilsion).all()

def convert_to_unit_vector(angles):
    yaw, pitch = angles[0], angles[1]

    x = -np.cos(yaw) * np.sin(pitch)
    y = -np.sin(yaw)
    z = -np.cos(pitch) * np.cos(pitch)
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    x = x / norm
    y = y / norm
    z = z / norm
    return [x, y, z]

def compute_angle_error(preds, labels):
    pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return np.arccos(angles) * 180 / np.pi

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]
ROOT_PATH = os.path.split(SCRIPT_PATH)[0]

def save_test_result(df):
    to_path = os.path.join(ROOT_PATH, "my_test_result.csv")
    df.to_csv(to_path, index=None)
    print("[GazeLib] Your test result has saved at {}".format(to_path))
