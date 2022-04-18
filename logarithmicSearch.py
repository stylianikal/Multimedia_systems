from createMacroblocks import createNeighbor
from sumOfAbsolutDifference import calculateSAD
import matplotlib.pyplot as plt

def logarithmicSearch(refFrame, targetMacroblock, indexMacroblock, k=16):
    macroblocksize = 16
    if k == 0:
        return indexMacroblock,\
               refFrame[indexMacroblock[0]:indexMacroblock[0] + macroblocksize,
               indexMacroblock[1]:indexMacroblock[1] + macroblocksize]

    refFrameNeighborMacroblocks = createNeighbor(refFrame, indexMacroblock, k)

    valuesSAD = calculateSAD(targetMacroblock, refFrameNeighborMacroblocks)
    indexMinSAD = divmod(valuesSAD.argmin(), valuesSAD.shape[1])
    newIndexMacroblock = list(indexMacroblock)

    if indexMinSAD[0] == 0:
        newIndexMacroblock[0] = indexMacroblock[0] - k
    elif indexMinSAD[0] == 2:
        newIndexMacroblock[0] = indexMacroblock[0] + k

    if indexMinSAD[1] == 0:
        newIndexMacroblock[1] = indexMacroblock[1] - k
    elif indexMinSAD[1] == 2:
        newIndexMacroblock[1] = indexMacroblock[1] + k

    if indexMinSAD[0] == 1 and indexMinSAD[1] == 1:
        newK = k // 2
    else:
        newK = k

    return logarithmicSearch(refFrame, targetMacroblock, tuple(newIndexMacroblock), newK)
