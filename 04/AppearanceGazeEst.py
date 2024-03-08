"""
All your implementations should be completed here.
"""

import pandas as pd
import numpy as np
from gazelib.utils import decode_base64_img
import copy

"""
Section 1: Dataset with pandas (35 points)

IMPORTANT: Make sure you have launched Jupyter notebook at 
section_1&2_gaze_estimation.ipynb

Read the guide there and complete the codes below simultaneously.
Pass local test at Jupyter notebook will help you a lot at online judge :).

"""


def mean_of_tgt_subject(df: pd.core.frame.DataFrame, subject_id: int) -> (float, float):
    """
    Section 1.1 - Compute mean yaw & pitch of that guy!  (5 points)

    A function that computes the average yaw and pitch for a subject

    :param df: a panda Dataframe, imported by load_train_csv_as_df(),
        contains columns of 'yaw', 'pitch', 'subject_id', etc..
    :param subject_id: an int, specifying target subject id

    :return: (yaw_mean, pitch_mean), a tuple of two floats, the mean of yaw and pitch of the target sbuject
    """

    ret = None
    df_mean = df.groupby('subject_id').mean()
    ret = (df_mean.loc[subject_id]['yaw'], df_mean.loc[subject_id]['pitch'])
    return ret


def count_tgt_subject(df: pd.core.frame.DataFrame, yaw_threshold: float) -> int:
    """
    Section 1.2 - Filter by yaw (5 points)

    A function that counts the number of images of which yaw is larger(>) than yaw_threshold

    :param df: a panda Dataframe, imported by load_train_csv_as_df(),
        contains columns of 'yaw', 'pitch', 'subject_id', etc..
    :param subject_id: a float

    :return: int, number of images that meets the requirement
    """

    ret = None

    df_cnt = df.loc[df['yaw'] > yaw_threshold]
    ret = len(df_cnt)

    return ret


def get_min_val_of_tgt_col(df: pd.core.frame.DataFrame, col: str):
    """
    Section 1.3 -  Get minimial value of target column (5 points)
    A function that gets the minimal value of target column

    :param df: a panda Dataframe, imported by load_train_csv_as_df(),
        contains columns of 'yaw', 'pitch', 'subject_id', etc..
    :param col, an string specifies which column we want

    :return: the minimial value of target column
    """

    ret = None

    df_min = df[col]
    ret = min(df_min)

    return ret


def compute_mean_eye(df: pd.core.frame.DataFrame) -> np.ndarray:
    """
    Section 1.4 - Mean eye is the perfect eye? (10 points)
    Mean eye denotes the image that takes average of all images(of the same size) w.r.t each pixel.

    Hint: for 'image_base64' column in dataframe, use decode_base64_img to decode it.

    :param df: a panda Dataframe, imported by load_train_csv_as_df(),
        contains columns of 'yaw', 'pitch', 'subject_id', etc..

    :return: a np.ndarry with dtype of np.uint8
    """

    ret = None

    df_meaneye = copy.deepcopy(df)
    ret = np.array(decode_base64_img(df_meaneye.loc[0, 'image_base64']), dtype=np.uint64)
    for i in range(1, len(df_meaneye)):
        ret = ret + np.array(decode_base64_img(df_meaneye.loc[i, 'image_base64']), dtype=np.uint64)
    ret = ret / len(df_meaneye)
    ret.astype(np.uint8)

    return ret


def add_glasses_info(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Section 1.5: Does he or she wear eyeglasses? (10 points)

    Add a column called 'has_glasses' with True/False to indicate whether he wears the glasses.
    The ids of subject with glasses have offered.

      - When subject id in `has_glasses_ids`, set the `has_glasses` value(a new  column) in df at that row as boolean varaible, True.
      - Otherwise, Set it to False.

    If you feel still confused, go to see the local judge at Jupyter noteboook

    :param df: a panda Dataframe, imported by load_train_csv_as_df(),
        contains columns of 'yaw', 'pitch', 'subject_id', etc..

    :return: a DataFrame with additional 'has_eyeglasses'
    """

    ret = None
    has_glasses_ids = [1, 4, 7, 9, 10]  # the subject ids with eye glasses
    ret = copy.deepcopy(df)

    ret['has_glasses'] = [False for i in range(len(ret))]
    for i in range(len(ret)):
        if ret.loc[i]['subject_id'] in has_glasses_ids:
            ret.loc[i, 'has_glasses'] = True
    return ret


"""
Section 2 - KNN with numpy (35 points)

In this section, we will build a gaze estimation system by KNN(K-nearest neighbour).

"""


def KNN_idxs(train_X, val_x, k=1):
    """
    Section 2.1 - K-top Eye Search Engine (10 points)

    Compute the euclidean distance between each sample in train_X with val_x
    :param train_X: a numpy array, [n, dim_n] or [n, W, H], of which each row is a sample, indicating the training set
    :param val_x: a numpy array, [dim_n] or [W, H], of which each row is a sample, indicating the validation sample
    :param k: an integer
    :return: a 1-D numpy array, indices of elements in train_X of which the order follows
    ascending euclidean distance between that sample and val_x
    """
    idxs = None

    dis = []
    idxs = []
    for train in train_X:
        dis.append(np.linalg.norm(train - val_x))
    dis = np.argsort(dis)
    for i in range(k):
        try:
            idxs.append(dis[i])
        except Exception:
            break
    return idxs


def oneNN(train_X, train_Y, val_x):
    """
    Section 2.2: From Search Engine To A Baseline Estimator - 1-NN estimator (10 points)

    1. Compute the Euclidean distance from the query example(val_x) to the labeled examples.
    2. Find the image with most similarity(one image in train_X with nearest euclidean distance).
    3. Take its corresponding label as output(train_Y[min_idx]).

    Hint: you could reuse your implementation of KNN_idxs(...), though sort is redundant for finding minimial

    :param train_X: a numpy array, [n, dim_n] or [n, W, H], of which each row is a sample, indicating the training set
    :param train_Y: a numpy array, [n, dim_n_out] of which each row is the corresponding label for each sample(in gaze context, it is gaze direction)
    :param val_x: a numpy array, [dim_n] or [W, H], of which each row is a sample, indicating the validation sample
    :return: a 1-d numpy array [dim_n_out,] or a number
    """
    ret = None

    ret = train_Y[KNN_idxs(train_X, val_x, 1)[0]]

    return ret


def KNN(train_X, train_Y, val_x, k=1):
    """
    Section 2.3: From  1-NN estimator to K-NN estimator (15 points)

    1.Compute the Euclidean distance(s) from the query example(val_x) to the labeled examples.
    2.Find the k images with most simlarity(k images in train_X with nearest euclidean distance)
    3.Take their median as output.

    Hint: you could reuse your implementation of KNN_idxs(...), though sort is redundant for finding k-minimal

    :param train_X: a numpy array, [n, dim_n] or [n, W, H], of which each row is a sample, indicating the training set
    :param train_Y: a numpy array, [n, dim_n_out] of which each row is the corresponding label for each sample(in gaze context, it is gaze direction)
    :param val_x: a numpy array, [dim_n] or [W, H], of which each row is a sample, indicating the validation sample
    :return: a 1-d numpy array [dim_n_out,] or a number
    """
    ret = None

    idxs = KNN_idxs(train_X, val_x, k)
    ret = []
    ret.append(train_Y[idxs[0]])
    for i in range(1, len(idxs)):
        ret.append(train_Y[idxs[i]])
    ret = np.array(ret)
    ret = np.median(ret, axis=0)

    return ret


"""
Section Bonus - Histogram of oriented gradient - numpy (+35 points)

Histogram of oriented gradient is widely used at computer vision community.

IMPORTANT: Make sure you have launched Jupyter notebook at 
section_1&2_gaze_estimation.ipynb
"""


def conv2d3x3(im: np.ndarray, kernel: np.ndarray):
    """
    Bonus-1: 2D Convolution - Learn how to compute with numpy (10 points)
    Implement the 2d convolution in this function (valid, no padding, stride=1)

    :param im: a numpy array, of size (H, W)
    :param kernel: a kernel of size (3, 3)
    :return:
    """
    assert im.shape[0] >= 3
    assert im.shape[1] >= 3

    ret = np.zeros((im.shape[0] - 2, im.shape[1] - 2), dtype=np.float)

    for i in range(im.shape[0] - 2):
        for j in range(im.shape[1] - 2):
            for k in range(3):
                for l in range(3):
                    ret[i][j] += im[i + k][j + l] * kernel[k][l]

    assert ret.shape[0] == im.shape[0] - 2
    assert ret.shape[1] == im.shape[1] - 2

    return ret


def conv2d3x3_same(im: np.ndarray, kernel: np.ndarray):
    """
    A padding policy(same) wrapper of conv2d 3x3 kernal based on your
    implementation of conv2d3x3 above (the valid padding version).

    :param im: a numpy array, of size (H, W)
    :param kernel: a kernel of size (3, 3)
    :return:
    """
    assert im.shape[0] >= 1
    assert im.shape[1] >= 1

    # to keep the same size, we add a padding of width 1
    ret = conv2d3x3(np.pad(im, 1, "reflect"), kernel)

    return ret


# Use kernels here to at compute_grad function keep the consistency with local/online judge
from gazelib.conv2d import gaussian_knl, sobel_hrztl, sobel_vtcl


def gaussian_kernel(im: np.ndarray):
    assert im.shape[0] >= 1
    assert im.shape[1] >= 1
    retx = conv2d3x3_same(conv2d3x3_same(im, gaussian_knl), sobel_hrztl)
    rety = conv2d3x3_same(conv2d3x3_same(im, gaussian_knl), sobel_vtcl)

    return (retx, rety)


def compute_grad(im: np.ndarray):
    """
    Bonus-2: Sobel operator (10 points) - Apply convolution by numpy
    Use conv2d3x3_same to compute the sobel gradient of the image:

    (1) Blur the image with the gaussian kernel(by conv2d3x3_same)
    (2) Compute gradient Gx(horizontal), Gy(vertical) by sobel_horizontal(Gx) & vertical kernel(Gy)
    (3) Stack Gx(at index 0), Gy(at index 1) together at axis 0.
    (4) Compute magnitude of gradient at each pixel.
        Hint: by applying np.linalg.norm to grad_dir
    (5) Derivie the normalized (norm-2) diretion vector
        For numerically robustness, when normalizing the gradient direction vector, add 1e-3 at denominator
        e.g: grad_dir_norm = grad_dir / (np.expand_dims(grad_mag, axis=0) + 1e-3)

    :param im: np.ndarray, with shape (H, W)
    :return: grad_dir_norm(the normalized gradient vector), grad_mag(the magnitude of the gradient)
    """
    gaze_dir_norm, gaze_mag = None, None
    Gx = None
    Gy = None

    ############## Your code here ##############
    # IMPORTANT: (1) to keep the same size, use conv2d3x3_same instead of conv2d3x3
    #                e.g: out = conv2d3x3_same(image, kernel)
    #            (2) to pass the offline/online check, use kernels above this function(gaussian_knl, ...)
    #            (3) For numerically robustness, when normalizing the gradient direction vector, add 1e-3 at denominator
    ############################################

    Gx, Gy = gaussian_kernel(im)[0], gaussian_kernel(im)[1]
    im_blur = im
    im_blur.shape = im.shape
    G = [[], []]
    G[0] = Gx.tolist()
    G[1] = Gy.tolist()
    grad_mag = np.zeros((Gx.shape[0], Gx.shape[1]), dtype=np.float)
    grad_dir_norm_x = [[0 for i in range(Gx.shape[1])] for j in range(Gx.shape[0])]
    grad_dir_norm_y = [[0 for i in range(Gx.shape[1])] for j in range(Gx.shape[0])]
    for i in range(Gx.shape[0]):
        for j in range(Gx.shape[1]):
            gx, gy = G[0][i][j], G[1][i][j]
            grad_dir = [gx, gy]
            gaze_mag = np.linalg.norm(grad_dir)
            grad_mag[i][j] = gaze_mag
            gaze_dir_norm = grad_dir / (np.expand_dims(gaze_mag, axis=0) + 1e-3)
            grad_dir_norm_x[i][j] = gaze_dir_norm[0]
            grad_dir_norm_y[i][j] = gaze_dir_norm[1]
    grad_dir_norm = np.array([grad_dir_norm_x, grad_dir_norm_y])

    assert grad_dir_norm.shape == (2, im_blur.shape[0], im_blur.shape[1])
    assert grad_mag.shape == im_blur.shape

    return grad_dir_norm, grad_mag


def bilinear_HOG_nonvec(grad_dir, grad_mag, bin_num=12):
    """
    Compute the Histogram for a patch.
    This is the reference code for next function(bilinear_HOG).
    All you need to do is to modify following codes to speed up this fuction.
    (a.k.a: replace built-in loops by numpy APIs)

    :param grad_dir: np.ndarray, with shape (2, W, H) of gradient at X-axis, Y-axis
    :param grad_mag: np.ndarray, with shape (W, H)
    :param bin_num: # bin
    :return:
    """
    ret_bin = np.zeros((bin_num,), dtype=np.float)
    bin_interval = np.pi * 2 / bin_num

    ############## Vectorization reference code ##############
    for i in range(grad_mag.shape[0]):
        for j in range(grad_mag.shape[1]):
            # arctan2 range: [-np.pi, np.pi]
            grad_dir_deg = np.arctan2(grad_dir[0, i, j], grad_dir[1, i, j]) + np.pi

            grad_bin_idx_l = int((grad_dir_deg) // bin_interval) % bin_num
            grad_bin_idx_r = int((grad_bin_idx_l + 1)) % bin_num

            grad_bin_lcoeff = grad_dir_deg / bin_interval - grad_bin_idx_l
            grad_bin_rcoeff = 1 - grad_bin_lcoeff

            ret_bin[grad_bin_idx_l] += grad_bin_lcoeff * grad_mag[i, j]
            ret_bin[grad_bin_idx_r] += grad_bin_rcoeff * grad_mag[i, j]
    #########################################################

    return ret_bin


def bilinear_HOG(grad_dir, grad_mag, bin_num=12):
    """
    Bonus-3: Re-implement HOG - learn vectorization at numpy(10 points)
    Compute the Histogram for a patch (Details refer to jupyter notebook)

    :param grad_dir: np.ndarray, with shape (2, W, H) of gradient at X-axis, Y-axis
    :param grad_mag: np.ndarray, with shape (W, H)
    :param bin_num: # bin
    :return:
    """
    ret_bin = np.zeros((bin_num,), dtype=np.float)
    bin_interval = np.pi * 2 / bin_num

    ############## Your code here #############
    # You should vectorize the bilinear_HOG_patch_nonvec here
    ###########################################
    grad_dir_deg = np.arctan2(grad_dir[0], grad_dir[1]) + np.pi
    grad_bin_idx_l = (grad_dir_deg // bin_interval) % bin_num
    grad_bin_idx_r = (grad_bin_idx_l + 1) % bin_num
    grad_bin_lcoeff = grad_dir_deg / bin_interval - grad_bin_idx_l
    grad_bin_rcoeff = 1 - grad_bin_lcoeff
    grad_bin_lcoeff = grad_bin_lcoeff * grad_mag
    grad_bin_rcoeff = grad_bin_rcoeff * grad_mag
    for i in range(grad_mag.shape[0]):
        for j in range(grad_mag.shape[1]):
            ret_bin[int(grad_bin_idx_l[i][j])] += grad_bin_lcoeff[i][j]
            ret_bin[int(grad_bin_idx_r[i][j])] += grad_bin_rcoeff[i][j]

    ###########################################

    return ret_bin


def bilinear_HOG_DB(im: np.ndarray, patch_num=(3, 4)):
    """
    Descriptor blocks - Learn how to index patches of image with numpy

    Note: Note: In this section, we give the implementation of descriptor blocks based on your
    previous function of HOG. If you do not complete the bonus-3, maually modify
    bilinear_HOG_DB function to use non-vectorization version of HOG.

    It uses the vectorization version by default.

    :param im: np.ndarray, with shape (H, W)
    :param patch_num: the nums of patch at Axis 0 and Axis 1
    :return:
    """
    grad_dir, grad_mag = compute_grad(im)

    patch_size = (30. / patch_num[0], 18. / patch_num[1])
    ret_feature = []

    for idx_h in range(patch_num[1]):
        for idx_w in range(patch_num[0]):
            patch_grad_dir = grad_dir[
                             :,
                             int(idx_h * patch_size[1]): int((idx_h + 1) * patch_size[1]),
                             int(idx_w * patch_size[0]): int((idx_w + 1) * patch_size[0])
                             ]

            patch_grad_mag = grad_mag[
                             int(idx_h * patch_size[1]): int((idx_h + 1) * patch_size[1]),
                             int(idx_w * patch_size[0]): int((idx_w + 1) * patch_size[0])
                             ]

            ret_feature.append(bilinear_HOG(patch_grad_dir, patch_grad_mag))
    ret_feature = np.concatenate(ret_feature)

    assert ret_feature.shape == (patch_num[0] * patch_num[1] * 12,)

    return ret_feature


def KNN_HOG(train_X, train_Y, val_x, k=1):
    """
    Bonus-4 Combine HOG with K-NN (5 points)

    1.Transform each image in train_X and val_x (2D) to HOG (1D)
    1.Compute the Euclidean distance(s) from the query example(val_x) to the labeled examples.
    2.Find the k samples with most simlarity(k HOGs in train_X with nearest euclidean distance)
    3.Take their median as output.

    Hint: you could reuse your implementation of KNN

    :param train_X: a numpy array, [n, W, H], of which each row is a sample, indicating the training set
    :param train_Y: a numpy array, [n, dim_n_out] of which each row is the corresponding label(gaze) for each sample(in gaze context, it is gaze direction)
    :param val_x: a numpy array, [W, H], of which each row is a sample, indicating the validation sample
    :return: a 1-d numpy array [dim_n_out,] or a number
    """
    ret = None

    val_x_hogged = bilinear_HOG_DB(val_x)
    train_X_hogged = []
    for train in train_X:
        train_X_hogged.append(bilinear_HOG_DB(train))
    train_X_hogged = np.array(train_X_hogged)
    ret = KNN(train_X_hogged, train_Y, val_x_hogged, k)

    return ret
