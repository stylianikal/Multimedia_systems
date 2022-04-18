import numpy as np


def SAD(referenceMacroblock, targetMacroblock):
    return np.sum(np.abs(targetMacroblock - referenceMacroblock))


def calculateSAD(targetMacroblock, referenceFrameNeighborMacroblocks):
    valuesSAD = []

    for macroblock in referenceFrameNeighborMacroblocks:
        if macroblock is not None:
            valuesSAD.append(SAD(macroblock, targetMacroblock))
        else:
            valuesSAD.append(np.Inf)

    return np.array(valuesSAD).reshape((3, 3))
