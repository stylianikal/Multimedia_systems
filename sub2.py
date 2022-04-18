from writeVideo import write
from splitVideoFrames import splitVideoFrames
from rgb2gray import rgb2gray
from motionCompensation import motionCompensation
from tqdm.autonotebook import tqdm
from createMacroblocks import imageReconstructFromBlocks
from framesDifference import framesDif
from entropy import averageEntropy
import matplotlib.pyplot as plt
import numpy as np


def runSub2(sample_path):
    videoFramesArray, fps = splitVideoFrames(sample_path)
    videoFramesSample = rgb2gray(videoFramesArray[:, :, :, :])

    compensatedVideoFramesSample = [videoFramesSample[0]]
    motionVectors = []
    for prevFrame, nextFrame in tqdm(zip(videoFramesSample, videoFramesSample[1:]), total=len(videoFramesSample[1:])):
        predict, vectors = motionCompensation(prevFrame, nextFrame)
        predFrame = imageReconstructFromBlocks(predict)
        compensatedVideoFramesSample.append(predFrame)
        motionVectors.append(vectors)

    compensatedFrameDifferenceSample = framesDif(
        np.array(compensatedVideoFramesSample).reshape(videoFramesSample.shape))

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(compensatedVideoFramesSample[1], cmap='gray')
    plt.title('N frame')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(compensatedVideoFramesSample[2], cmap='gray')
    plt.title('N+1 frame')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(compensatedFrameDifferenceSample[1], cmap='gray')
    plt.title('N+1 - N frame')
    plt.show()

    write(compensatedVideoFramesSample, '../auxiliary2021/compensatedSample.mp4', fps)
    write(compensatedFrameDifferenceSample, '../auxiliary2021/compensatedErrorSample.mp4', fps)

    print("Project 1_2")
    print("Average Entropy: ", averageEntropy(compensatedVideoFramesSample, videoFramesSample))
