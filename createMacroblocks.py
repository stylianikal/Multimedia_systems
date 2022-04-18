import numpy as np


def createMacroblocks(frame):
    macroblocksize = 16
    macroblocks = []
    x, y = frame.shape
    for i in range(0, x, macroblocksize):
        for j in range(0, y, macroblocksize):
            macroblock = frame[i:i + macroblocksize, j:j + macroblocksize]
            if macroblock.shape == (macroblocksize, macroblocksize):
                macroblocks.append(macroblock)
            else:
                try:
                    macroblock = np.vstack(
                        (macroblock, np.zeros(macroblock.shape[0], macroblocksize - macroblock.shape[1])))
                except TypeError:
                    pass
                try:
                    macroblock = np.hstack(
                        (macroblock, np.zeros(macroblocksize - macroblock.shape[0], macroblock.shape[1])))
                except TypeError:
                    pass
                macroblocks.append(macroblock)

    return np.array(macroblocks).reshape(
        (int(x / macroblocksize), int(y / macroblocksize), macroblocksize, macroblocksize))


def imageReconstructFromBlocks(blocks):
    lines = []
    for i in range(blocks.shape[0]):
        line = []
        for j in range(blocks.shape[1]):
            line.append(blocks[i, j, :, :])
        line = np.hstack(line)
        lines.append(line)

    return np.vstack(lines)


def createNeighbor(referenceFrame, indexOfMacroblock, k=16):
    macroblocksize = 16
    neighbor = []
    for i in range(indexOfMacroblock[0] - k, indexOfMacroblock[0] + k + 1, k):
        for j in range(indexOfMacroblock[1] - k, indexOfMacroblock[1] + k + 1, k):
            if (i >= 0 and j >= 0 and i + macroblocksize < referenceFrame.shape[0] and j + macroblocksize <
                    referenceFrame.shape[1]):
                neighbor.append(referenceFrame[i:i + macroblocksize, j:j + macroblocksize])
            else:
                neighbor += [None]

    return neighbor