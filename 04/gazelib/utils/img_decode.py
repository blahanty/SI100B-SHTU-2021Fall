import base64
from io import BytesIO
from PIL import Image
import numpy as np

resize_size = [30, 18]

def decode_base64_img(base64_str: str) -> np.ndarray:
    im = base64.b64decode(base64_str.encode("ascii"))
    im = BytesIO(im)
    im = Image.open(im).resize(resize_size)

    return np.array(im)
