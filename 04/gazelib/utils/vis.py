import numpy as np
import matplotlib.pyplot as plt


def vis_yaw_pitch(df):
    plt.figure(figsize=(6, 3))
    plt.subplot(121)
    plt.title("Yaw (Unit: deg.)")
    plt.hist(df['yaw'] / np.pi * 180, bins=30)
    plt.subplot(122)
    plt.title("Pitch (Unit: deg.)")
    plt.hist(df['pitch'] / np.pi * 180, bins=30)
    plt.show()

def vis_grad(raw_im, Gx, Gy):
    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(231)
    plt.title("Raw Image")
    plt.imshow(raw_im, cmap="gray", vmin=0, vmax=1)
    plt.colorbar()

    plt.subplot(232)
    plt.title("Gradient X")
    plt.imshow(raw_im, cmap="gray", vmin=0, vmax=1)
    plt.imshow(Gx, vmin=-1, vmax=1, cmap="RdBu", alpha=0.7)
    plt.colorbar()

    plt.subplot(233)
    plt.title("Gradient Y")
    plt.imshow(raw_im, cmap="gray", vmin=0, vmax=1)
    plt.imshow(Gy, vmin=-1, vmax=1, cmap="RdBu", alpha=0.7)
    plt.colorbar()

    plt.subplot(223)
    mag = np.linalg.norm(np.array([Gx, Gy]), axis=0)
    plt.title("Gradient magnitude")
    plt.imshow(mag, vmin=0, vmax=1)
    plt.colorbar()


    plt.subplot(224)
    plt.title("Gradient")
    plt.imshow(raw_im, cmap="gray", vmin=0, vmax=1)
    plt.colorbar()

    step = 3
    margin = 2
    len_mul = 4.0

    for x in range(margin, raw_im.shape[1] - margin, step):
        for y in range(margin, raw_im.shape[0] - margin, step):
            plt.arrow(x, y, Gx[y, x] * len_mul, Gy[y, x] * len_mul, length_includes_head=True,
                  head_width=0.8, head_length=0.8, color="red")


    plt.show()

def vis_HOG(raw_im, Gx, Gy, hist):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(221)
    plt.title("Raw Image")
    plt.imshow(raw_im, cmap="gray", vmin=0, vmax=1)
    plt.colorbar()


    plt.subplot(222)
    plt.title("Gradient")
    plt.imshow(raw_im, cmap="gray", vmin=0, vmax=1)
    plt.colorbar()

    step = 2
    margin = 2
    len_mul = 4.0

    for x in range(margin, raw_im.shape[1] - margin, step):
        for y in range(margin, raw_im.shape[0] - margin, step):
            c = np.array([Gx[y, x], Gy[y, x], 0.4])
            margin_c = 0.1
            c = (c + 1) / 2
            c = c * (1 - margin_c * 2) + margin_c
            plt.arrow(x, y, Gx[y, x] * len_mul, Gy[y, x] * len_mul, length_includes_head=True,
                  head_width=0.8, head_length=0.8, color=c, alpha=0.8)

    plt.subplot(212)
    plt.stem([i for i in range(-180, 180, 30)], hist)
    plt.title("Histogram of oriented gradients(in one patch)")
    plt.xlabel("Gradient Degree(Unit: deg.)")
    plt.ylabel("Gradient maginitude(Unit: Pixel)")


    plt.show()

def vis_HOG_full(raw_im, Gx, Gy, hist, patch_num=(3, 4), bin_num=12):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(221)
    plt.title("Raw Image")
    plt.imshow(raw_im, cmap="gray", vmin=0, vmax=1)

    ylines = np.linspace(0, raw_im.shape[0] - 0.5, patch_num[0] + 1)
    xlines = np.linspace(0, raw_im.shape[1] - 0.5, patch_num[1] + 1)

    for y in ylines:
        plt.axhline(y=y, ls=":", c="red", alpha=0.8, linewidth=3)
    for x in xlines:
        plt.axvline(x=x, ls=":", c="red", alpha=0.8, linewidth=3)
    plt.colorbar()


    plt.subplot(222)
    plt.title("Gradient")
    plt.imshow(raw_im, cmap="gray", vmin=0, vmax=1)
    plt.colorbar()

    step = 2
    margin = 2
    len_mul = 4.0

    for x in range(margin, raw_im.shape[1] - margin, step):
        for y in range(margin, raw_im.shape[0] - margin, step):
            c = np.array([Gx[y, x], Gy[y, x], 0.4])
            margin_c = 0.1
            c = (c + 1) / 2
            c = c * (1 - margin_c * 2) + margin_c
            plt.arrow(x, y, Gx[y, x] * len_mul, Gy[y, x] * len_mul, length_includes_head=True,
                  head_width=0.8, head_length=0.8, color=c, alpha=0.8)

    plt.subplot(212)
    for i in range(0, hist.shape[0] + 1, 12):
        plt.axvline(x=i, ls=":", c="red", alpha=0.5)
    plt.stem([i for i in range(hist.shape[0])], hist)
    plt.title("Histogram of oriented gradients")
    plt.xlabel("Features")
    plt.ylabel("Gradient maginitude(Unit: Pixel)")


    plt.show()

def convert_gaze_vect(yp):
    yaw, pitch = yp[0], yp[1]
    y = -np.sin(pitch)
    proj_zox = np.cos(pitch)

    z = -np.cos(yaw) * proj_zox
    x = -np.sin(yaw) * proj_zox

    return np.array([x, y, z])

def visualize_est(im, gaze_est, gaze_gt):
    length = 25

    xyz_est = convert_gaze_vect(gaze_est)
    xyz_gt = convert_gaze_vect(gaze_gt)

    x_center, y_center = 30 // 2, 18 // 2

    plt.imshow(im, cmap="gray", vmin=0, vmax=1)
    plt.plot(
        [x_center, x_center + xyz_gt[0] * length],
        [y_center, y_center + xyz_gt[1] * length],
        linewidth=5, c="r", label="ground truth", alpha=0.5
    )
    plt.plot(
        [x_center, x_center + xyz_est[0] * length],
        [y_center, y_center + xyz_est[1] * length],
        linewidth=5, c="b", label="estimation", alpha=0.5
    )
    plt.legend()

def visualize_est_test(im, gaze_est):
    length = 25

    xyz_est = convert_gaze_vect(gaze_est)

    x_center, y_center = 30 // 2, 18 // 2

    plt.imshow(im, cmap="gray", vmin=0, vmax=1)
    plt.plot(
        [x_center, x_center + xyz_est[0] * length],
        [y_center, y_center + xyz_est[1] * length],
        linewidth=5, c="b", label="estimation", alpha=0.8
    )
    plt.legend()

def visualize_angle_loss(train_loss, val_loss, print_interval):
    plt.figure(figsize=[8, 4])
    x = [i for i in range(0, len(train_loss) * print_interval, print_interval)]

    plt.scatter(x, train_loss)
    plt.scatter(x, val_loss)

    plt.plot(x, train_loss, label="Train loss")
    plt.plot(x, val_loss, label="Val loss")

    plt.legend()
    plt.xlabel("# iter.")
    plt.ylabel("Angle error(Unit: deg.)")

    plt.show()

