import imageio


def write(sample, savePath, fps):
    writer = imageio.get_writer(savePath, fps=fps, mode="I")

    for frame in sample:
        writer.append_data(frame)
    writer.close()