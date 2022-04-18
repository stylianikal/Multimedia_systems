import numpy as np


def framesDif(frames):
    diflist = []
    for i in range(1, frames.shape[0]):
        diflist.append(frames[i, :, :] - frames[i - 1, :, :])
    return np.array(diflist)