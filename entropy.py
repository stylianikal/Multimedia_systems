import numpy as np
from scipy.stats import entropy


def entropyImage(img):
    _, num = np.unique(img, return_counts=True)
    return entropy(num, base=2)


def averageEntropy(videoFramesSample, shape):
    avgEntropy = 0
    for frame in videoFramesSample:
        avgEntropy += entropyImage(frame)

    avgEntropy /= shape.shape[0]
    return avgEntropy
