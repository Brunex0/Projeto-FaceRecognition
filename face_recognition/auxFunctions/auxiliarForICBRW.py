import cv2
import numpy as np

def readPairs(cfgData):
    """
        Read all pairs from icbrw pairs file.
        :param cfgData: The config data
        :return: returns the matched and mismatched pairs
    """
    pairsFile = open(cfgData['pairs-path'], 'r')
    linesFromFile = pairsFile.readlines()
    matchedPairs=[]
    mismatchedPairs = []
    for line in linesFromFile:
        str = line.split('\t')
        if '1' in str[2]:
            matchedPairs.append([str[0], str[1]])
        elif '0' in str[2]:
            mismatchedPairs.append([str[0], str[1]])

    return matchedPairs, mismatchedPairs


