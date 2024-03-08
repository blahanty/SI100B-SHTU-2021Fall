import os
from .utils import load_obj
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]
MEAN_EYE_PATH = os.path.join(SCRIPT_PATH, "mean_eye.pkl")

def assert_eq(val_a, val_b, epilsion=1e-3):
    """

    :param val_a:
    :param val_b:
    :param epilsion:
    :return:
    """
    assert abs(val_a - val_b) < epilsion

def check_im_similarity(ipt_mean_eye: np.ndarray):
    gt_mean_eye = load_obj(MEAN_EYE_PATH)
    diff = (ipt_mean_eye.astype(np.float64) - gt_mean_eye.astype(np.float64)) / 255
    is_almost_same = (np.linalg.norm(diff) / diff.size <= 1e-3).all()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Your mean eye")
    plt.imshow(ipt_mean_eye, vmin=0, vmax=256, cmap="gray")
    plt.subplot(122)
    plt.title("Ground truth eye")
    plt.imshow(gt_mean_eye, vmin=0, vmax=256, cmap="gray")

    if is_almost_same:
        print('You pass the local test - Section 1.5 (5%)')
    else:
        raise ValueError('Your image is not the same as the ground truth image')
