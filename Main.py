from splitVideoFrames import splitVideoFrames
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from motionCompensation import motionCompensation
from createMacroblocks import imageReconstructFromBlocks
from rgb2gray import rgb2gray
import numpy as np
from writeVideo import write

sample_path = "../auxiliary2021/sample2.mp4"
videoFramesArray, fps = splitVideoFrames(sample_path)
videoFramesSample = rgb2gray(videoFramesArray[:, :, :, :])

compensatedVideoFramesSample = [videoFramesSample[0]]
motionVectors = []
for prevFrame, nextFrame in tqdm(zip(videoFramesSample, videoFramesSample[1:]), total=len(videoFramesSample[1:])):
    predict, vectors = motionCompensation(prevFrame, nextFrame)
    predFrame = imageReconstructFromBlocks(predict)
    compensatedVideoFramesSample.append(predFrame)
    motionVectors.append(vectors)
    plt.imshow(predFrame, cmap='gray')
    plt.show()

write(np.array(compensatedVideoFramesSample).reshape(videoFramesSample.shape), '../auxiliary2021/compensatedRemoveObject.mp4', fps)

plt.imshow(videoFramesSample[0], cmap='gray')
plt.show()