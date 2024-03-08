import numpy as np

def assert_eq_np(val_a, val_b, epilsion=1e-3):
    """

    :param val_a:
    :param val_b:
    :param epilsion:
    :return:
    """
    assert (np.abs(val_a - val_b) < epilsion).all()

def assert_eq_2223(grad_dir, grad_mag):

    gt_grad_dir_x = np.array(
        [[0.,         0.99990245, 0.],
         [0.,         0.95925723, 0.],
         [0.,         0.99985187, 0.]]
    )
    gt_grad_dir_y = np.array(
        [[0.,         0.,         0.],
         [0.99976476, 0.28213448, 0.99866844],
         [0.,        0.,         0.]]
    )
    gt_grad_dir = np.array([gt_grad_dir_x, gt_grad_dir_y])

    gt_grad_mag = np.array([
        [ 0.,         10.25,        0.],
        [ 4.25,        8.86002257,  0.75],
        [ 0.,         6.75,        0.]
    ])
    assert_eq_np(gt_grad_dir, grad_dir)
    assert_eq_np(gt_grad_mag, grad_mag)
