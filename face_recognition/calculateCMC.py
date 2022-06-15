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
from tensorflow.keras.applications.resnet50 import preprocess_input

import seaborn as sns

def prevision(cfgData, model, url):
    img = cv2.imread(url)
    img = cv2.resize(img, (cfgData['inputSize'], cfgData['inputSize']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    embeds1 = model(img)
    if cfgData['test-type'] == 'Align-L2' or cfgData['test-type'] == 'Align-L2-CosineSim':
        embeds1 = K.l2_normalize(embeds1, axis=1)

    return embeds1


def saveIdentities(cfgData, model, n_classes):
    # create labels
    labels = []
    labels2 = []
    Identities = []
    for i in range(1, n_classes + 1):
        labels.append(str(i).zfill(3))
        labels2.append(str(i - 1))

    for i in range(0, n_classes):
        url = "E:\\Projeto-FaceRecognition\\face_recognition\\dataset\\icbrw_ProbeImages_mtcnn_224\\" + labels2[i] + "\\" + labels[i] + "_01.png"
        #url = "E:\\Projeto-FaceRecognition\\face_recognition\\dataset\\icbrw_GalleryImages_mtcnn_224\\" + labels2[i] + "\\" + labels[i] + "_f.png"
        print(url)
        X = prevision(cfgData, model, url)
        Identities.append(X.numpy())

    data = np.asarray(Identities)
    np.save('Identities.npy', data)


def nearestFace(X):
    dist = []
    data = np.load('Identities.npy')
    for i in range(0, 90):
        tensor = tf.convert_to_tensor(data[i], dtype=tf.float32)
        dist.append(metrics.pairwise.cosine_similarity(tensor, X)[0][0])
        # dist.append(-metrics.pairwise.euclidean_distances(tensor, X)[0][0])
    return dist


if "__main__":
    cfgData = load_yaml('config.yml')
    folders = glob(cfgData['train-path'] + "/*")
    modelTest = ResNet50WithSoftmax(cfgData, 10575)
    modelTest.load_weights(cfgData['model-weights-path'])

    model_part = tf.keras.Model(
        inputs=modelTest.input,
        outputs=modelTest.get_layer("flatten").output
    )

    #saveIdentities(cfgData,model_part, 90)
    distances = []
    for i in range(0, 90):
        for j in range(1, 6):
            url = "E:\\Projeto-FaceRecognition\\face_recognition\\dataset\\icbrw_ProbeImages_mtcnn_224\\" + str(i) + "\\" + str(i + 1).zfill(3) + "_" + str(j).zfill(2) + ".png"
            # url2 = "E:\\Projeto-FaceRecognition\\face_recognition\\dataset\\icbrw_GalleryImages_mtcnn_224\\" + str(i) + "\\" + str(i+1).zfill(3) + "_f.png"
            print(url)
            # X = prevision(cfgData, model, url)
            vector = nearestFace(prevision(cfgData, model_part, url))
            """print(vector)
            print(min(vector))
            print(vector.index(min(vector)))
            print('----------------')"""
            vector1 = np.argsort(vector)[::-1]
            #print(vector1)
            # np.squeeze(vector)
            distances.append(vector1)

    matrix = []
    print(len(distances))

    id = 0
    count=0
    for value in distances:
        idLocation = np.where(value == id)[0][0]
        array = np.zeros(90, dtype=int)
        array[idLocation] = 1
        matrix.append(array)
        count+=1
        if count % 5 == 0:
            id+=1

    rank=[]
    rank.append(np.array(matrix)[:, 0])
    for i in range(1,90):
        rank.append(np.array(matrix)[:, i])
        rank[i] = rank[i] + rank[i-1]
    import matplotlib.pyplot as plt
    graph=[]
    for item in rank:
        graph.append(sum(item)/450)

    sns.set_style('darkgrid')  # darkgrid, white grid, dark, white and ticks
    sns.color_palette('pastel')
    plt.rc('axes', titlesize=18)  # fontsize of the axes title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=13)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
    plt.rc('legend', fontsize=13)  # legend fontsize
    plt.rc('font', size=13)  # controls default text sizes
    plt.figure(figsize=(7, 7), tight_layout=True)
    plt.plot(range(1, len(graph) + 1), graph)
    plt.xlabel('Rank')
    plt.ylabel('Identification Accuracy')
    plt.title('CMC')
    plt.savefig('E:\\Projeto-FaceRecognition\\face_recognition\\graphs\\CMCCurve.png')

    print("rank1", graph[0])
    print("rank2", graph[1])
    print("rank3", graph[2])
    print("rank4", graph[3])
    print("rank5", graph[4])
    print("rank10", graph[9])
