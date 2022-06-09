import os.path

import tensorflow as tf
import numpy as np
from model import ResNet50WithSoftmax
from glob import glob
from loadContentFiles import load_yaml
from auxFunctions.auxiliarForLFW import *
from auxFunctions.auxiliarForICBRW import *
from auxFunctions.auxiliarForROC import *
import matplotlib.pyplot as plt
from datetime import datetime

if "__main__":
    cfgData = load_yaml('config.yml')
    folders = glob(cfgData['train-path'] + "/*")
    modelTest = ResNet50WithSoftmax(cfgData, 10580)
    modelTest.load_weights(cfgData['model-weights-path'])
    #model.load_weights(ckpt_path)
    model_part = tf.keras.Model(
        inputs=modelTest.input,
        outputs=modelTest.get_layer("flatten").output
    )
    checkpoint_timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    pathForSave = os.path.join('evaluationsResult', checkpoint_timestamp, cfgData['dataset-name'])
    if not os.path.exists(pathForSave):
        os.makedirs(pathForSave)

    if cfgData['dataset-name'] == 'LFW':
        numMatchedPairs, matchedPairs, numMisMatchedPairs, mismatchedPairs = createPairsArray(cfgData)
        TPR, FPR, auc = calculateRocAndAccuracy(cfgData, model_part, matchedPairs, mismatchedPairs)
        np.savez(pathForSave + '/'+cfgData['test-type']+'.npz', x=TPR, y=FPR, z=auc)
    elif cfgData['dataset-name'] == 'ICBRW':
        matchedPairs, mismatchedPairs = readPairs(cfgData)
        TPR, FPR, auc = calculateRocAndAccuracy(cfgData, model_part, matchedPairs, mismatchedPairs)
        np.savez(pathForSave + '/' + cfgData['test-type'] + '.npz', x=TPR, y=FPR, z=auc)
        print(auc)
        plt.plot(FPR, TPR)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

#nvidia-smi ->ver percentagem da GPU