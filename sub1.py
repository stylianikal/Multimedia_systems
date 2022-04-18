import matplotlib.pyplot as plt
from splitVideoFrames import splitVideoFrames
from rgb2gray import rgb2gray
from entropy import averageEntropy
from framesDifference import framesDif
from writeVideo import write

def runSub1(sample_path):
    videoFramesArray, fps = splitVideoFrames(sample_path)
    videoFramesSample = videoFramesArray[:, :, :, :]
    videoFramesSample = rgb2gray(videoFramesSample)
    frameDifferenceSample = framesDif(videoFramesSample)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(videoFramesSample[1], cmap='gray')
    plt.title('N frame')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(videoFramesSample[2], cmap='gray')
    plt.title('N+1 frame')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(frameDifferenceSample[1], cmap='gray')
    plt.title('N+1 - N frame')
    plt.show()

    write(frameDifferenceSample, '../auxiliary2021/errorSeqSample.mp4', fps)

    print("Project 1_1")
    print("Average Entropy: ", averageEntropy(videoFramesSample, videoFramesSample))
