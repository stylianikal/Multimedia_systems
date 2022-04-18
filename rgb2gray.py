import numpy as np


def rgb2gray(sample):
    framesGrayscale = []
    for i in sample:
        framesGrayscale.append(np.around(np.dot(i[:, :, :3], [0.2989, 0.587, 0.114])))

    return np.array(framesGrayscale)