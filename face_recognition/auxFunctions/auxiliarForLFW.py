import cv2
import numpy as np

def createPairsArray(cfgData):
    file = open(cfgData['pairs-path'], 'r')
    lines = file.readlines()
    matchedPairs = []
    numMatchedPairs = 0
    mismatchedPairs = []
    numMisMatchedPairs = 0
    for line in lines:
        str = line.split('\t')
        if len(str) == 3:
            img1 = cfgData['test-path'] + str[0] + '/' + str[0] + '_' + str[1].zfill(4) + '.png'
            img2 = cfgData['test-path'] + str[0] + '/' + str[0] + '_' + str[2].rstrip().zfill(4) + '.png'
            matchedPairs.append([img1, img2])
            numMatchedPairs += 1
        elif len(str) == 4:
            img1 = cfgData['test-path'] + str[0] + '/' + str[0] + '_' + str[1].zfill(4) + '.png'
            img2 = cfgData['test-path'] + str[2] + '/' + str[2] + '_' + str[3].rstrip().zfill(4) + '.png'
            mismatchedPairs.append([img1, img2])
            numMisMatchedPairs += 1
    return numMatchedPairs, matchedPairs, numMisMatchedPairs, mismatchedPairs


