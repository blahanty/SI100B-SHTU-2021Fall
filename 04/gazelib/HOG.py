import numpy as np

def bilinear_HOG_patch_nonvec(grad_dir, grad_mag, bin_num=12):
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
