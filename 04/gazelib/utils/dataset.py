import requests
import time
import os
import hashlib
import pandas as pd
from PIL import Image

__all__ = ['download_csv_mpIIdataset', 'load_train_csv_as_df', 'load_test_csv_as_df']

SCRIPT_PATH = os.path.realpath(__file__)
ROOT_PATH = os.path.split(os.path.split(os.path.split(SCRIPT_PATH)[0])[0])[0]
DATASET_PATH = os.path.join(ROOT_PATH, "dataset")

TRAIN_CSV_URL = "http://data.liubai01.cn:81/f/768e16137c1e4fb1b1c6/?dl=1"
TEST_CSV_URL = "http://data.liubai01.cn:81/f/0d94a125c8c24f059254/?dl=1"
DEMO_IM_URL = "http://data.liubai01.cn:81/f/5c59f03d303740cfa3ce/?dl=1"

TRAIN_MD5 = '67a8f558ff50ccb8b0df0f57b79d0690'
TEST_MD5 = '513ccc363f40b2d048bf52c339f8cb53'
DEMO_IM_MD5 = 'c142123b048ad392af53908bfd046e46'

def load_train_csv_as_df():
    train_csv_path = os.path.join(DATASET_PATH, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError("Train csv not found, you are required to download it from cells above!")
    return pd.read_csv(train_csv_path)

def load_test_csv_as_df():
    test_csv_path = os.path.join(DATASET_PATH, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError("Test csv not found, you are required to download it from cells above!")
    return pd.read_csv(test_csv_path)

def getmd5(file):
    """
    Source: https://blog.csdn.net/zheng_ruiguo/article/details/88717711
    :param file:
    :return:
    """
    m = hashlib.md5()
    with open(file,'rb') as f:
        for line in f:
            m.update(line)
    md5code = m.hexdigest()
    return md5code

def _download_url_to_path(to_path: str, url: str) -> bool:
    """
    Modified from: https://blog.csdn.net/weixin_30624825/article/details/97345151
    :param to_path:
    :param url:
    :return: bool, whether download succeeds or not
    """
    try:
        filename = os.path.split(to_path)[-1]
        headers = {'Proxy-Connection': 'keep-alive'}
        r = requests.get(url, stream=True, headers=headers)
        length = float(r.headers['content-length'])

        print(filename + ': ' + 'Start downloading', end='\r')

        with open(to_path, 'wb') as f:
            count = 0
            count_tmp = 0

            timer = time.time()
            for chunk in r.iter_content(chunk_size = 512):
                if chunk:
                    f.write(chunk)
                    count += len(chunk)
                    if time.time() - timer > 2:
                        p = count / length * 100
                        speed = (count - count_tmp) / 1024 / 1024 / 2
                        count_tmp = count
                        print(filename + ': ' + '{:.2f}'.format(p) + '%' + ' Speed: ' + '{:.2f}'.format(speed) + 'M/S', end='\r')
                        timer = time.time()
        print()
    except Exception as e:
        print()
        print(e)
        return False
    return True

def download_csv_mpIIdataset():
    """Already Downloaded"""
    
    """
    This function download MPIIGaze dataset to the assignment_Gaze/dataset diretory.
    Note that it is a modified csv version of the MPIIGaze, for the educational purpose of this assignment.

    ==================================================
    The official raw dataset project link:

    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/
    research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/

    @inproceedings{zhang15_cvpr,
    Author = {Xucong Zhang and Yusuke Sugano and Mario Fritz and Bulling, Andreas},
    Title = {Appearance-based Gaze Estimation in the Wild},
    Booktitle = {Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    Year = {2015},
    Month = {June}
    Pages = {4511-4520}
    }
    ==================================================

    :return:
    """
    train_to_path = os.path.join(DATASET_PATH, "train.csv")
    test_to_path = os.path.join(DATASET_PATH, "test.csv")

    if os.path.exists(train_to_path) and getmd5(train_to_path) == TRAIN_MD5:
        print("[GazeLib] Train csv exists and passed md5 cheking.")
    else:
        raise ValueError("train csv error")

    if os.path.exists(test_to_path) and getmd5(test_to_path) == TEST_MD5:
        print("[GazeLib] Test csv exists and passed md5 cheking.")
    else:
        raise ValueError("test csv error")

def download_demo_Img():
    """Already Downloaded"""
    demo_Img_path = os.path.join(DATASET_PATH, "demo.jpeg")
    if os.path.exists(demo_Img_path) and getmd5(demo_Img_path) == DEMO_IM_MD5:
        print("[GazeLib] Demo image exists and passed md5 cheking.")
    else:
        raise ValueError("demo img error")

def load_demo_Img():
    filenames = os.listdir(DATASET_PATH)
    imgnames = [filename for filename in filenames if not filename.endswith(".csv") and not filename.endswith(".md")]
    return Image.open(os.path.join(DATASET_PATH, imgnames[0]))
