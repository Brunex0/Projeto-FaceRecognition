import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras.backend as K
from sklearn import metrics

def calculateDistance(cfgData, model, pair1, pair2):
    """
        Calculate the distance of two pairs.
        :param cfgData: The config data
        :param model: The model to predict the features
        :param pair1: The first element of a pair
        :param pair2: The second element of a pair
        :return: returns the distance
    """
    label = np.zeros(90)
    img = cv2.imread(pair1)

    img = cv2.resize(img, (cfgData['inputSize'], cfgData['inputSize']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)

    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    embeds1 = model(img)
    if cfgData['test-type'] == 'Align-L2' or cfgData['test-type'] == 'Align-L2-CosineSim':
        embeds1 = K.l2_normalize(embeds1, axis=1)

    img = cv2.imread(pair2)
    img = cv2.resize(img, (cfgData['inputSize'], cfgData['inputSize']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)

    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    embeds2 = model(img)
    if cfgData['test-type'] == 'Align-L2' or cfgData['test-type'] == 'Align-L2-CosineSim':
        embeds2 = K.l2_normalize(embeds2, axis=1)

    if cfgData['test-type'] == 'Align-L2-CosineSim':
        return metrics.pairwise.cosine_similarity(embeds1, embeds2)[0]
    else:
        return -metrics.pairwise.euclidean_distances(embeds1, embeds2)[0]


def calculateRocAndAccuracy(cfgData, model, matchPairs, mismatchPairs):
    """
        Calculate the Roc and accuracy.
        :param cfgData: The config data
        :param model: The model to predict the features
        :param matchPairs: The pairs with images of the same person
        :param mismatchPairs: The pairs with images of the different persons
        :return: returns the TPR (True positive rate), FPR (False positive rate), AUC (Area Under The Curve), ACC (Accuracy)
    """

    y_score = []
    y_true = []

    print("Matched Pairs")
    for pair in matchPairs:
        y_true.append(1)
        y_score.append(calculateDistance(cfgData, model, pair[0], pair[1]))

    print("Mismatched Pairs")
    for pair in mismatchPairs:
        y_true.append(0)
        y_score.append(calculateDistance(cfgData, model, pair[0], pair[1]))

    print("ROC curve inicializer")
    y_score = [1.0 * number for number in y_score]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)

    auc = metrics.auc(fpr, tpr)

    opt_idx = np.argmax(tpr - fpr)
    opt_threshold = thresholds[opt_idx]
    y_pred = [1 if s >= opt_threshold else 0 for s in y_score]
    acc = metrics.accuracy_score(y_true, y_pred)

    return tpr, fpr, auc, acc