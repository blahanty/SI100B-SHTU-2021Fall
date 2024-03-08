import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]
HAARC_PATH = os.path.join(SCRIPT_PATH, "haarcascades")

def to_equal_histo(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    return cdf[image]


def estimate_gaze(image, eyeim2gaze):
    raw_image = image
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2BGRA)

    face_detector = cv2.CascadeClassifier(os.path.join(HAARC_PATH, 'haarcascade_frontalface_default.xml'))
    eye_detector = cv2.CascadeClassifier(os.path.join(HAARC_PATH, 'haarcascade_eye.xml'))

    face_zone = face_detector.detectMultiScale(gray,1.3,3,minSize=(80,80))

    x, y, w, h = face_zone[0]

    h_up = int(face_zone[0,-1] * 0.6)

    head = gray[y: y + h,x: x + w]
    head_up = head[0: h_up]

    eye_zone = eye_detector.detectMultiScale(head_up, 1.3, 3, minSize=(10, 10))
    left_eye, right_eye = None, None

    ex_0, ex_1 = eye_zone[0][0], eye_zone[1][0]

    right_eye_idx = int(ex_0 > ex_1)

    for idx in range(2):
        ex, ey, ew, eh = eye_zone[idx]
        image_eye = gray[ey + y: ey + eh + y, ex + x: ex + ew + x]
        image_eye = cv2.cvtColor(image_eye,cv2.COLOR_RGB2BGR)

        w_e, h_e, c_e = image_eye.shape
        w_to = float(h_e) / 30 * 18
        w_start = int((float(w_e) - w_to) // 2)
        w_end = int(w_start + w_to)
        image_eye = image_eye[w_start: w_end]
        image_eye = cv2.cvtColor(image_eye, cv2.COLOR_RGB2GRAY)
        image_eye = cv2.resize(image_eye, (30, 18))
        image_eye = np.array(image_eye)
        image_eye = to_equal_histo(image_eye) / 256

        if idx == right_eye_idx:
            right_eye = image_eye
            eye_x_right = ex + x + ew // 2
            eye_y_right = ey + y + ey // 2
        else:
            left_eye = image_eye
            eye_x_left = ex + x + ew // 2
            eye_y_left = ey + y + ey // 2

    left_gaze = eyeim2gaze(left_eye)
    right_gaze = eyeim2gaze(right_eye[:, ::-1])
    right_gaze[0] = -right_gaze[0]
    length = 120


    plt.figure(figsize=(6, 6))
    plt.imshow(raw_image)


    plt.scatter([eye_x_left], [eye_y_left], c="r")
    plt.scatter([eye_x_right], [eye_y_right], c="b")

    plt.plot(
        [x, x, x + w, x + w, x],
        [y, y + h, y + h, y, y],
        label="face", linewidth=3, c='black'
    )
    plt.plot(
        [eye_x_left, eye_x_left + left_gaze[0] * length],
        [eye_y_left, eye_y_left + left_gaze[1] * length],
        linewidth=4, c="r", label="left gaze", alpha=0.7
    )


    plt.plot(
        [eye_x_right, eye_x_right + right_gaze[0] * length],
        [eye_y_right, eye_y_right + right_gaze[1] * length],
        linewidth=4, c="b", label="right gaze", alpha=0.7
    )
    plt.legend()
    plt.show()


    plt.figure(figsize=(6, 6))
    ax3d = plt.gca(projection='3d')

    ax3d.scatter([0], [eye_x_left], [eye_y_left], c="r")
    ax3d.scatter([0], [eye_x_right], [eye_y_right], c="b")

    ax3d.plot(
        [0, left_gaze[2] * length],
        [eye_x_left, eye_x_left + left_gaze[0] * length],
        [eye_y_left, eye_y_left + left_gaze[1] * length],
        label="left gaze", linewidth=3, c="r"
    )

    ax3d.plot(
        [0, right_gaze[2] * length],
        [eye_x_right, eye_x_right + right_gaze[0] * length],
        [eye_y_right, eye_y_right + right_gaze[1] * length],
         label="right gaze", linewidth=3, c="b"
    )

    ax3d.plot(
        [0, 0, 0, 0, 0],
        [x, x, x + w, x + w, x],
        [y, y + h, y + h, y, y],
        label="face", linewidth=3, c='black'
    )


    ax3d.set_ylim(0, image.shape[1])
    ax3d.set_ylabel("Image horizontal line")
    ax3d.set_zlim(image.shape[0], 0)
    ax3d.set_zlabel("Image vertical line")

    ax3d.set_xlim(-length * 1.5, 0)

    plt.legend()
    plt.tight_layout()
    plt.show()
