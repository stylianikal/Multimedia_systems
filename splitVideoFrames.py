import imageio
import numpy as np


def splitVideoFrames(file):
    video = imageio.get_reader(file, 'ffmpeg')
    fps = video.get_meta_data()['fps']
    i = 0
    frames = []
    while 1:
        try:
            img = video.get_data(i)
            frames.append(img)
            i += 1
        except IndexError:
            break

    return np.array(frames), fps