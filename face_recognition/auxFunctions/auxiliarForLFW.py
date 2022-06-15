import cv2
import numpy as np

def createPairsArray(cfgData):
    """
        Read all pairs from icbrw pairs file and format for the correct path.
        :param cfgData: The config data
        :return: returns the matched and mismatched pairs
    """
    file = open(cfgData['pairs-path'], 'r')
    lines = file.readlines()
    matchedPairs = []
    mismatchedPairs = []
    for line in lines:
        str = line.split('\t')
        if len(str) == 3:
            img1 = cfgData['test-path'] + str[0] + '/' + str[0] + '_' + str[1].zfill(4) + '.png'
            img2 = cfgData['test-path'] + str[0] + '/' + str[0] + '_' + str[2].rstrip().zfill(4) + '.png'
            matchedPairs.append([img1, img2])
        elif len(str) == 4:
            img1 = cfgData['test-path'] + str[0] + '/' + str[0] + '_' + str[1].zfill(4) + '.png'
            img2 = cfgData['test-path'] + str[2] + '/' + str[2] + '_' + str[3].rstrip().zfill(4) + '.png'
            mismatchedPairs.append([img1, img2])
    return matchedPairs, mismatchedPairs


