import tensorflow as tf
from model import model
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn import metrics

pairs = '/home/socialab/Joao/projects/face_recognition_softmax/lfw_funneled/pairs_forTestInTrainning.txt'
simpleDataset = '/home/socialab/Joao/projects/face_recognition_softmax/lfw_aligned_mtcnn_224/'


def createPairsArray(url, pairsList):
    file = open(pairsList, 'r')
    lines = file.readlines()
    matchedPairs = []
    numMatchedPairs = 0
    mismatchedPairs = []
    numMisMatchedPairs = 0
    for line in lines:
        str = line.split('\t')

        if len(str) == 3:
            img1 = url + str[0] + '/' + str[0] + '_' + str[1].zfill(4) + '.png'
            img2 = url + str[0] + '/' + str[0] + '_' + str[2].rstrip().zfill(4) + '.png'
            matchedPairs.append([img1, img2])
            numMatchedPairs += 1
        elif len(str) == 4:
            img1 = url + str[0] + '/' + str[0] + '_' + str[1].zfill(4) + '.png'
            img2 = url + str[2] + '/' + str[2] + '_' + str[3].rstrip().zfill(4) + '.png'
            mismatchedPairs.append([img1, img2])
            numMisMatchedPairs += 1
    return numMatchedPairs, matchedPairs, numMisMatchedPairs, mismatchedPairs


def calculateDistance(model, pair1, pair2):

    img = cv2.imread(pair1)
    img = cv2.resize(img, (112, 112))
    # img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    # img = img / 255.

    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    embeds1 = model(img)
    embeds1 = K.l2_normalize(embeds1, axis=1)
    
    img = cv2.imread(pair2)
    img = cv2.resize(img, (112, 112))
    # img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    # img = img / 255.

    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    embeds2 = model(img)
    embeds2 = K.l2_normalize(embeds2, axis=1)
    
    return metrics.pairwise.cosine_similarity(embeds1, embeds2)[0]
    #return tf.norm(embeds1 - embeds2, ord='euclidean')


def calculateRocAndAccuracy(model, matchPairs, mismatchPairs):

    y_score = []
    y_true = []
    TruePR = []
    FalsePR = []
    valuesForAcc = []
    acc = 0
    print("Start Computing distances")
    
    print("Matched Pairs")
    for pair in matchPairs:
        y_true.append(1)
        y_score.append(calculateDistance(model, pair[0], pair[1]))
        
    print("Mismatched Pairs")
    for pair in mismatchPairs:
        y_true.append(0)
        y_score.append(calculateDistance(model, pair[0], pair[1]))
        
    print("ROC curve inicializer")
    y_score = [1.0 * number for number in y_score]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)

    auc = metrics.auc(fpr, tpr)
    
    return tpr, fpr, auc


def lfw_eval_callback(model):


    numMatchedPairs, matchedPairs, numMisMatchedPairs, mismatchedPairs = createPairsArray(simpleDataset, pairs)

    model_part = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("flatten").output)

    TPR, FPR, auc = calculateRocAndAccuracy(model_part, matchedPairs, mismatchedPairs)
    print(auc)
