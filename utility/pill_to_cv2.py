from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image

def imgread(filestr):
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    # resize the array (image) then PIL image
    im_resized = im_pil.resize((180, 180))
    return im_resized